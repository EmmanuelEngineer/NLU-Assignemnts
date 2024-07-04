import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
from pprint import pprint


class MultiModel1(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1,
                 pad_index=0, bidirectional=False,
                 emb_dropout=0.1, n_layers=1, dropout=False, model_type="LSTM"):
        super(MultiModel1, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        assert model_type in [
            "LSTM", "BERT"], (f"Model Type {model_type} not Recognized")
        self.lstm_model = (model_type == "LSTM")
        self.dropout_flag = dropout
        if model_type == "LSTM":
            self.embedding = nn.Embedding(
                vocab_len, emb_size, padding_idx=pad_index)
        if (dropout):
            self.drop1 = nn.Dropout(emb_dropout)
        if model_type == "LSTM":
            self.utt_encoder = nn.LSTM(
                emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)
        elif model_type == "BERT":
            self.utt_encoder = BertModel.from_pretrained("bert-base-uncased")
        # we have 2 linear layers because we have 2 tasks with different layers
        if bidirectional and model_type == "LSTM":
            output_hidden_dim = hid_size*2*n_layers
        else:
            output_hidden_dim = hid_size*n_layers
        self.slot_out = nn.Linear(output_hidden_dim, out_slot)
        # The LSTM works as the encoder
        self.intent_out = nn.Linear(output_hidden_dim, out_int)

    def forward(self, utterance, seq_lengths, attention_mask = None):
        # utterance.size() = batch_size X seq_len
        # we are vectorizing the inputs
        # utt_emb.size() = batch_size X seq_len X emb_size
        if (self.lstm_model):
            utt_emb = self.embedding(utterance)

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        if self.lstm_model and self.dropout_flag:
            utt_emb = self.drop1(utt_emb)
        if self.lstm_model:
            packed_input = pack_padded_sequence(
                utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        else:
            packed_input = pack_padded_sequence(
                utterance, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence the encoder is the LSTM and we give it each time a token and see the hidden state for each token
        utt_encoded, input_sizes = pad_packed_sequence(
            packed_output)
        # Get the last hidden state
        if self.lstm_model and self.utt_encoder.bidirectional:
            last_hidden = torch.cat(
                (last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=1)
        else:
            last_hidden = last_hidden[-1, :, :]

        # Is this another possible way to get the last hiddent state? (Why?)
        # we use the last token since the network has seen the all sequence before the last
        # utt_encoded.permute(1,0,2)[-1]

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    


class Bertized(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, vocab_len,
                    pad_index=0,
                    emb_dropout=0.1, dropout=False, model_type="LSTM"):
        super(Bertized, self).__init__()
        self.dropout_flag = dropout
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.slot_out = nn.Linear(768, out_slot)
        # The LSTM works as the encoder
        self.intent_out = nn.Linear(768, out_int)

    def forward(self, attention_mask, utterance, token_type_ids):
        outputs = self.bert(attention_mask=attention_mask, input_ids=utterance, token_type_ids=token_type_ids )
        last_hidden_states = outputs.last_hidden_state
        #last_hidden_states = last_hidden_states[-1, :, :]
        slots = self.slot_out(last_hidden_states[:,0,:])
        intent = self.intent_out(last_hidden_states[:,1:, :])
        return slots, intent
