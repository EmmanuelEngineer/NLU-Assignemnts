import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.optim.optimizer import Optimizer, required


class V_Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.__p = p
        self.__mask = None

    def generate_mask(self, x):
        self.__mask = torch.rand(*x.shape, device='cuda:0') <= self.__p

    def get_mask(self):
        return self.__mask

    def forward(self, x):
        if self.training:
            x = x * self.__mask
        return x


class MultiModel(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, model_type, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, dropout_type="None", Weight_tying=False):
        super(MultiModel, self).__init__()
        self.dropout_layer = dropout_type

        assert (model_type in ["RNN", "LSTM"]), model_type + " not compatible"
        if (model_type == "RNN"):
            model_core = nn.RNN
        elif (model_type == "LSTM"):
            model_core = nn.LSTM

        assert (dropout_type in ["None", "Variational",
                "Normal"]), dropout_type + "not compatible"
        if (dropout_type == "Normal"):
            dropout_layer = nn.Dropout
        elif (dropout_type == "Variational"):
            dropout_layer = V_Dropout

        # Token ids to vectors, we will better see this in the next lab
        # matrix vocabulary*dimension of the input
        self.embedding = nn.Embedding(
            output_size, emb_size, padding_idx=pad_index)

        if (dropout_type != "None"):
            self.drop1 = dropout_layer(p=emb_dropout)
        self.core_nn = model_core(emb_size, hidden_size,
                                  n_layers, bidirectional=False)
        self.pad_token = pad_index

        if (dropout_type != "None"):
            self.drop2 = dropout_layer(p=out_dropout)

        # Linear layer to project the hidden layer to our output space
        if Weight_tying:
            assert (
                emb_size == hidden_size), "For Weight Tying emb_size and hidden size must be the same"
            self.output = nn.Linear(hidden_size, output_size)
            self.output.weight.data = self.embedding.weight.data
        else:
            self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        x = self.embedding(input_sequence)  # sentence
        if (self.dropout_layer != "None"):
            x = self.drop1(x)
        x, _ = self.core_nn(x)
        if (self.dropout_layer != "None"):
            x = self.drop2(x)
        output = self.output(x).permute(0, 2, 1)
        return output

    def reset_dropout(self, input_sample):
        assert (self.dropout_layer in ["Variational",
                "Normal"]), "reset dropout should not activate"
        emb = self.embedding(input_sample)
        self.drop1.generate_mask(emb.to("cuda:0"))
        drop1_out = self.drop1(emb)
        lstm_out, _ = self.core_nn(drop1_out)
        self.drop2.generate_mask(lstm_out)


class NTAvSGD(optim.SGD):
    def __init__(
            self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(NTAvSGD, self).__init__(
            params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov
        )
        self.avg_params = None

    def initialize_avg_params(self):
        self.avg_params = []
        for group in self.param_groups:
            avg_group = {}
            for param in group["params"]:
                if param.requires_grad:
                    avg_group[param] = torch.clone(param.data).detach()
            self.avg_params.append(avg_group)

    def update_avg_params(self):
        for avg_group, group in zip(self.avg_params, self.param_groups):
            for param in group["params"]:
                if param.requires_grad:
                    avg_group[param].data.mul_(0.5).add_(param.data, alpha=0.5)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Standard SGD step
        super().step()

        # Initialize avg_params on first step
        if self.avg_params is None:
            self.initialize_avg_params()

        self.update_avg_params()
        self.swap_params_with_avg()
        self.flatten_rnn_parameters()

    def swap_params_with_avg(self):
        if self.avg_params is not None:
            for avg_group, group in zip(self.avg_params, self.param_groups):
                for param in group["params"]:
                    if param.requires_grad:
                        param.data, avg_group[param].data = (
                            avg_group[param].data,
                            param.data,
                        )

    def flatten_rnn_parameters(self):
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, torch.nn.RNNBase):
                    param.flatten_parameters()

    def state_dict(self):
        state_dict = super(NTAvSGD, self).state_dict()
        state_dict["avg_params"] = [
            {k: v.clone() for k, v in avg_group.items()}
            for avg_group in self.avg_params
        ]
        return state_dict

    def load_state_dict(self, state_dict):
        avg_params = state_dict.pop("avg_params")
        self.avg_params = [
            {k: v.clone() for k, v in avg_group.items()} for avg_group in avg_params
        ]
        super(NTAvSGD, self).load_state_dict(state_dict)


class AverageOfGradientsSGD(optim.SGD):

    def __init__(self, params, lr=1e-3, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False):
        super(AverageOfGradientsSGD, self).__init__(
            params, lr, momentum, dampening, weight_decay, nesterov
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'avg_grad' not in state:
                    state['avg_grad'] = torch.zeros_like(p.data)
                    state['step'] = 0

                avg_grad = state['avg_grad']
                step = state['step']

                # Update average gradient
                avg_grad.mul_(step).add_(p.grad).div_(step + 1)
                state['step'] += 1

                # Update parameters after avg_steps
                if state['step']:
                    p.data.add_(-lr, avg_grad)
                    state['step'] = 0
                    state['avg_grad'].zero_()

        return loss
