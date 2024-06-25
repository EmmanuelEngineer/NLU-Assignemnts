import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from functools import partial
import torch.utils.data as data
import utils as ut
import math
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from model import AverageOfGradientsSGD
from model import V_Dropout
from model import NTAvSGD
from model import MultiModel
import spacy
import itertools
import numpy as np
DEVICE = 'cuda:0'  # it can be changed with 'cpu' if you do not have a gpu
nlp = spacy.load('en_core_web_lg')
EPOCHS = 2


# RNN Elman version
# We are not going to use this since for efficiency purposes it's better to use the RNN layer provided by pytorch


class RNN_cell(nn.Module):
    def __init__(self,  hidden_size, input_size, output_size, vocab_size, dropout=0.1):
        super(RNN_cell, self).__init__()

        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, prev_hidden, word):
        input_emb = self.W(word)
        prev_hidden_rep = self.U(prev_hidden)
        # ht = σ(Wx + Uht-1 + b)
        hidden_state = self.sigmoid(input_emb + prev_hidden_rep)
        # yt = σ(Vht + b)
        output = self.output(hidden_state)
        return hidden_state, output
# Loading the corpus


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids


def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")


# Vocab is computed only on training set
# We add two special tokens end of sentence and padding
# The dataset was already cutoffed
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])


# This class computes and stores our vocab
# Word to ids and ids to word
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output


lang = Lang(train_raw, ["<pad>", "<eos>"])


class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            # We get from the first token till the second-last token
            self.source.append(sentence.split()[0:-1])
            # We get from the second token till the last token
            self.target.append(sentence.split()[1:])
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    # Map sequences of tokens to corresponding computed in Lang class
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(
                            param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)


def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(
            len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        # We remove these tensors from the computational graph
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item


# Dataloader instantiation
# You can reduce the batch_size if the GPU memory is not enough
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=partial(
    collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(
    collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(
    collate_fn, pad_token=lang.word2id["<pad>"]))


# Tensor board logging system


def log_values(writer, step, ppl, prefix):
    writer.add_scalar(f"{prefix}/ppl", ppl, step)


def avg_train_loop(data, optimizer, criterion, model, VDROP, clip=5):
    torch.cuda.empty_cache()
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:  # data is the list of batches
        optimizer.zero_grad()  # Zeroing the gradient

        if VDROP:
            model.reset_dropout(sample["source"])
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, zeroing the computational graph calculated by pytorch
        # clip the gradient to avoid explosive gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights    weight_decay_range, patience_range, emb_drop_range,

    return sum(loss_array)/sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            # torch.cuda.empty_cache()
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def train_model(
        param_string,
    n_epochs=100,
    patience_set=3,
    losses_train=[],
    losses_dev=[],
    sampled_epochs=[],
    best_ppl=math.inf,
    best_model=None,
    modelM=None,
    optimizer=None,
    NM_ASGD=False,
    ASGD_optim=NTAvSGD,
    Logging_interval=1,
    non_monotone_interval=5,
    lr=.5,
    VDROP=False,
    clip=5,
    criterion_train=None,
    weight_decay_Av=.05,
    criterion_eval=None,
    exp_name=None,
):

    patience = patience_set
    writer = SummaryWriter(log_dir=f"runs/{param_string}")
    pbar = tqdm(range(1, n_epochs))
    # If the PPL is too high try to change the learning rate
    loss_log = []
    T = 0
    t = 0
    for epoch in pbar:
        loss = avg_train_loop(train_loader, optimizer,
                              criterion_train, modelM, VDROP, clip)
        # The epoch is treated as the k in NT-AvSGD
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, modelM)
            losses_dev.append(np.asarray(loss_dev).mean())
            # 4-9: of NT-AvSGD
            if NM_ASGD and epoch % Logging_interval == 0 and T == 0:
                if (len(loss_log) > 0):
                    if t > non_monotone_interval and ppl_dev > min(loss_log):
                        T = epoch
                        optimizer = ASGD_optim(
                            modelM.parameters(), lr=lr, weight_decay=weight_decay_Av)
                        print(f"swiching to ASGD at epoch{epoch}")
                        patience = patience_set
                loss_log.append(ppl_dev)
                t = t+1

            # tensorboard logger
            log_values(writer, epoch, ppl_dev, exp_name+"PPL")

            pbar.set_description("PPL: %f Patience:%d" % (ppl_dev, patience))

            if ppl_dev < best_ppl:  # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(modelM).to('cpu')
                patience = patience_set
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    return final_ppl, modelM


# Don't forget to experiment with a lower training batch size
# Increasing the back propagation steps can be seen as a regularization step

# With SGD try with an higher learning rate (> 1 for instance)
vocab_len = len(lang.word2id)

criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(
    ignore_index=lang.word2id["<pad>"], reduction='sum')


def train_grid_search(lr_range, clip_range, hid_size_range, emb_size_range, weight_decay_range, patience_range, emb_drop_range, out_drop_range, vocab_len, model_type="None", dropout_type="None", NM_ASGD=False, optimizer=None, n_epochs=100, exp_name="Pollo", ASGD_optim=NTAvSGD, Weight_tying=False):
    best_model = None
    best_accuracy = 99999
    best_params = {}
    optimizer_class = optimizer
    model = None
    # Iterate over all combinations of the parameter ranges
    for lr, clip, hid_size, emb_size, weight_decay, patience, emb_drop, out_drop in itertools.product(lr_range, clip_range, hid_size_range, emb_size_range, weight_decay_range, patience_range, emb_drop_range, out_drop_range):
        if Weight_tying and emb_drop != out_drop:
            continue

        print(
            f"Training with parameters: lr={lr}, clip={clip}, hid_size={hid_size}, emb_size={emb_size}, weight_decay={weight_decay}, patience={patience}, emb_drop={emb_drop}, out_drop={out_drop}")

        # Initialize the model with current parameters
        model = MultiModel(emb_size, hid_size, vocab_len, model_type=model_type,
                           pad_index=lang.word2id["<pad>"], dropout_type=dropout_type,
                           emb_dropout=emb_drop, out_dropout=out_drop,
                           Weight_tying=Weight_tying).to(DEVICE)

        model.apply(init_weights)

        # Set up the optimizer with current parameters
        optimizer_obj = optimizer_class(model.parameters(), lr=lr,
                                        weight_decay=weight_decay)

        if dropout_type == "Variational":
            VDROP = True
        else:
            VDROP = False
        # Train the model
        accuracy, model = train_model(exp_name+"/"+f"LR={lr}, C={clip}, HS={hid_size}, ES={emb_size}, WD={weight_decay}, P={patience}, E_D={emb_drop}, O_D={out_drop},M_T={model_type},D_T={dropout_type},ASGD={NM_ASGD}, VDROP={VDROP},WT={Weight_tying}", clip=clip, modelM=model,
                                      optimizer=optimizer_obj, NM_ASGD=NM_ASGD, VDROP=VDROP,
                                      criterion_eval=criterion_eval,
                                      criterion_train=criterion_train, patience_set=patience, weight_decay_Av=weight_decay,
                                      n_epochs=n_epochs, exp_name=exp_name,
                                      ASGD_optim=ASGD_optim)

        # Check if the current model is the best so far
        if accuracy < best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = {
                'lr': lr,
                'clip': clip,
                'hid_size': hid_size,
                'emb_size': emb_size,
                'weight_decay': weight_decay,
                'patience': patience,
                'emb_drop': emb_drop,
                'out_drop': out_drop
            }

            torch.save({
                'model_state_dict': model.state_dict(),
            }, "models/"+exp_name+".model")

    print(f"Best model parameters: {best_params}")
    print(f"Best model accuracy: {best_accuracy}")
    return best_model, best_params


Lr = [1.0, 2.0, 3.0]
Clip = [1, 5, 10]
Hid_size = [100, 250, 500]
Emb_size = [100, 250, 500]
Weighy_decay = [0.0001, 0.001]
Patience_range = [3]
Emb_drop = [0]
Out_drop = [0]

best_model, best_params = train_grid_search(
    Lr, Clip, Hid_size, Emb_size,
    Weighy_decay, Patience_range, Emb_drop,
    Out_drop, vocab_len, exp_name="LM_RNN_SGD", model_type="RNN", optimizer=optim.SGD, dropout_type="None", n_epochs=EPOCHS)


Lr = [.02, .002, .001]
Clip = [1, 5, 10]
Hid_size = [100, 250, 500]
Emb_size = [100, 250, 500]
Weighy_decay = [0.0001, 0.001]
Patience_range = [3]
Emb_drop = [0]
Out_drop = [0]


best_model, best_params = train_grid_search(
    Lr, Clip, Hid_size, Emb_size,
    Weighy_decay, Patience_range, Emb_drop,
    Out_drop, vocab_len, exp_name="LSTM_SIMPLE_AdamW", model_type="LSTM", optimizer=optim.AdamW, dropout_type="Normal", n_epochs=EPOCHS)

Lr = [.02, .002, .001]
Clip = [1, 5, 10]
Hid_size = [100, 250, 500]
Emb_size = [100, 250, 500]
Weighy_decay = [0.0001, 0.001]
Patience_range = [3]
Emb_drop = [0.1, 0.2, 0.3]
Out_drop = [0.1, 0.2, 0.3]

best_model, best_params = train_grid_search(
    Lr, Clip, Hid_size, Emb_size,
    Weighy_decay, Patience_range, Emb_drop,
    Out_drop, vocab_len, exp_name="LSTM_DROP_AdamW", model_type="LSTM", optimizer=optim.AdamW, dropout_type="None", n_epochs=EPOCHS)


Lr = [1.0, 2.0, 3.0]
Clip = [1, 5, 10]
Hid_size = [100, 250, 500]
Emb_size = [100, 250, 500]
Weighy_decay = [0.0001, 0.001]
Patience_range = [3]
Emb_drop = [0.1, 0.2, 0.3]
Out_drop = [0.1, 0.2, 0.3]

best_model, best_params = train_grid_search(
    Lr, Clip, Hid_size, Emb_size,
    Weighy_decay, Patience_range, Emb_drop,
    Out_drop, vocab_len, exp_name="LSTM_SIMPLE_NT-AvSGD",
    optimizer=optim.SGD, dropout_type="None", model_type="LSTM", n_epochs=EPOCHS, NM_ASGD=True,
    ASGD_optim=AverageOfGradientsSGD)

Lr = [1.0, 2.0, 3.0]
Clip = [1, 5, 10]
Hid_size = [100, 250, 500]
Emb_size = [100, 250, 500]
Weighy_decay = [0.0001, 0.001]
Patience_range = [3]
Emb_drop = [0.1, 0.2, 0.3]
Out_drop = [0.1, 0.2, 0.3]

best_model, best_params = train_grid_search(
    Lr, Clip, Hid_size, Emb_size,
    Weighy_decay, Patience_range, Emb_drop,
    Out_drop, vocab_len,  exp_name="LSTM_SIMPLE_NTASGD_VDROP",
    optimizer=optim.SGD, dropout_type="Variational", model_type="LSTM", n_epochs=EPOCHS, NM_ASGD=True)


Lr = [1.0, 2.0, 3.0]
Clip = [1, 5, 10]
Hid_size = [100, 250, 500]
Emb_size = [100, 250, 500]
Weighy_decay = [0.0001, 0.001]
Patience_range = [3]
Emb_drop = [0.1, 0.2, 0.3]
Out_drop = [0.1, 0.2, 0.3]

best_model, best_params = train_grid_search(
    Lr, Clip, Hid_size, Emb_size,
    Weighy_decay, Patience_range, Emb_drop,
    Out_drop, vocab_len,  exp_name="LSTM_NTASGD_VDROP_WEIGHT",
    optimizer=optim.SGD, dropout_type="Variational", model_type="LSTM", n_epochs=EPOCHS, NM_ASGD=True, Weight_tying=True)
