import torch
import torch.nn as nn

from transformers import BertConfig, BertModel
from tokenizers import Tokenizer

import random
from data import pad_var_sequences

class SmallBertModel(nn.Module):
    def __init__(
        self,
        tokenizers: dict,
        vocab_size: int = 20000,
        max_position_embeddings: int = 100,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        **args,
    ):
        super(SmallBertModel, self).__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )
        self.bert = BertModel(config)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tokenizer_zh :Tokenizer = tokenizers['zh']
        self.tokenizer_en :Tokenizer = tokenizers['en']

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **args):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_scores = self.linear(outputs.last_hidden_state)
        return prediction_scores
    

def build_bert_model(**args):
    model = SmallBertModel(**args)
    return model

class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act=nn.ReLU()):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.act = act

    def forward(self, x):
        return self.act(self.fc(x))


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional)

    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell): # the decoder run one step at a time, the input (1, B, H)
        output, (hidden, cell) = self.lstm(input.unsqueeze(0), (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

class SeqLSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, bidirectional_encoder=True, num_layers=1, device="cuda"):
        super(SeqLSTM, self).__init__()
        self.in_emb = nn.Embedding(vocab_size, input_size)
        self.out_emb = nn.Embedding(vocab_size, hidden_size)
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, bidirectional_encoder)
        self.decoder = LSTMDecoder(vocab_size, hidden_size, num_layers)
        self.softmax = nn.LogSoftmax(dim=1)
        if bidirectional_encoder:
            t_hidden_size = 2 * hidden_size
        self.h_t = DenseLayer(t_hidden_size, hidden_size, nn.Tanh())
        self.c_t = DenseLayer(t_hidden_size, hidden_size, nn.Tanh())
        self.vocab_size = vocab_size
        self.device = device
        

    def forward(self, src, trg, lengths, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.vocab_size

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        src = self.in_emb(src)
        padded_src = pad_var_sequences(src, lengths)
        hidden, cell = self.encoder(padded_src)

        if self.encoder.bidirectional:
            hidden = hidden.view(self.encoder.num_layers, -1, self.encoder.hidden_dim * 2)
            cell = cell.view(self.encoder.num_layers, -1, self.encoder.hidden_dim * 2)

        hidden = self.h_t(hidden)
        cell = self.c_t(cell)

        input = trg[0, :]

        for t in range(1, trg_len):
            input = self.out_emb(input)
            prediction, hidden, cell = self.decoder(input, hidden, cell)
            prob = self.softmax(prediction)
            outputs[t] = prediction
            top1 = prob.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs
