import torch
import torch.nn as nn

from transformers import BertConfig, BertModel
from tokenizers import Tokenizer

from data import pad_var_sequences, pack_var_sequences, sequence_mask

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
    
class AttnLayer(nn.Module):
    def __init__(self, act="softmax"):
        super(AttnLayer, self).__init__()
        if act == "softmax":
            self.act = nn.Softmax(dim=1)

    def forward(self, enc_out, dec_out, enc_mask=None):
        # for easily using in training, do not squeeze the dim
        if dec_out.dim() == 2:
            dec_out = dec_out.unsqueeze(2) # (B, H, 1)
        elif dec_out.dim() == 3:
            dec_out = dec_out.permute(0, 2, 1) # (B, H, S)
        attn = torch.bmm(enc_out, dec_out) # (B, L, S) or (B, L, 1)
        if enc_mask is not None:
            enc_mask = enc_mask.unsqueeze(2).expand(-1, -1, attn.shape[2]) # (B, L, S)
            attn = attn.masked_fill(enc_mask.bool(), -1e9)
        attn = self.act(attn)
        attn = attn.permute(0, 2, 1) # (B, S, L)
        attn_out = torch.bmm(attn, enc_out) # (B, S, H)
        r"""
        another way to calculate the attn_out
        but note that bmm is twice faster than einsum in this simple case

        attn_out = torch.einsum("bls,blh->bsh", attn, enc_out) # (B, S, H)
        """
        return attn_out, attn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return outputs, hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, input, hidden, cell): # the decoder run one step at a time, the input (B, 1, H)
        output, (hidden, cell) = self.lstm(input.unsqueeze(1), (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell
    
class LSTMCellDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(LSTMCellDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, input, hidden, cell): # input (B, H), output (B, O)
        hidden, cell = self.lstm(input, (hidden, cell))
        prediction = self.fc(hidden)
        return prediction, hidden, cell


class SeqLSTMv1(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, bidirectional_encoder=True, num_layers=1, act=False, device="cuda"):
        super(SeqLSTMv1, self).__init__()
        self.in_emb = nn.Embedding(vocab_size, input_size)
        self.out_emb = nn.Embedding(vocab_size, hidden_size)
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, bidirectional_encoder)
        self.decoder = LSTMDecoder(vocab_size, hidden_size, num_layers)
        self.softmax = nn.LogSoftmax(dim=1)
        if bidirectional_encoder:
            t_hidden_size = 2 * hidden_size
        if act:
            self.h_t = DenseLayer(t_hidden_size, hidden_size, nn.Tanh())
            self.c_t = DenseLayer(t_hidden_size, hidden_size, nn.Tanh())
        else:
            self.h_t = nn.Linear(t_hidden_size, hidden_size, bias=False)
            self.c_t = nn.Linear(t_hidden_size, hidden_size, bias=False)
        self.vocab_size = vocab_size
        self.device = device
    
    def forward(self, src, src_len, trg, trg_len, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0] # src with shape (B, L)
        trg_len_size = trg.shape[1]
        trg_vocab_size = self.vocab_size

        outputs = torch.zeros(batch_size, trg_len_size, trg_vocab_size).to(self.device)

        src = self.in_emb(src)
        padded_src = pad_var_sequences(src, src_len)
        _, hidden, cell = self.encoder(padded_src)

        if self.encoder.bidirectional:
            hidden = hidden.view(self.encoder.num_layers, -1, self.encoder.hidden_dim * 2)
            cell = cell.view(self.encoder.num_layers, -1, self.encoder.hidden_dim * 2)

        hidden = self.h_t(hidden)
        cell = self.c_t(cell)

        input = trg[:, 0]

        if teacher_forcing_ratio == 1: # use teacher forcing
            teaching_trg = trg[:, :-1]
            padded_trg = pad_var_sequences(self.out_emb(teaching_trg), trg_len[:-1])
            padded_pred, hidden, cell = self.decoder(padded_trg, hidden, cell)
            pred = pack_var_sequences(padded_pred, total_length=trg_len_size)
            outputs[:, 1:, :] = pred
            return outputs

        random_teacher_forcing = torch.rand(trg_len_size - 1, device=self.device) < teacher_forcing_ratio
        for t in range(1, trg_len_size):
            input = self.out_emb(input)
            prediction, hidden, cell = self.decoder(input, hidden, cell)
            prob = self.softmax(prediction)
            outputs[:, t, :] = prediction
            top1 = prob.argmax(1)
            input = trg[:, t] if random_teacher_forcing[t-1] else top1

        return outputs


class SeqLSTMv2(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, bidirectional_encoder=True, num_layers=1, device="cuda"):
        super(SeqLSTMv2, self).__init__()
        self.in_emb = nn.Embedding(vocab_size, input_size)
        self.out_emb = nn.Embedding(vocab_size, hidden_size)
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, bidirectional_encoder)
        self.decoder = LSTMCellDecoder(hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)
        if bidirectional_encoder:
            t_hidden_size = 2 * hidden_size
        self.h_t = nn.Linear(t_hidden_size, hidden_size, bias=False)
        self.c_t = nn.Linear(t_hidden_size, hidden_size, bias=False)
        self.o_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.device = device

    def forward(self, src, src_len, trg, trg_len, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len_size = trg.shape[1]
        trg_vocab_size = self.vocab_size

        outputs = torch.zeros(batch_size, trg_len_size, trg_vocab_size).to(self.device)

        src = self.in_emb(src)
        padded_src = pad_var_sequences(src, src_len)
        _, hidden, cell = self.encoder(padded_src)

        if self.encoder.bidirectional:
            hidden = hidden.view(self.encoder.num_layers, -1, self.encoder.hidden_dim * 2)
            cell = cell.view(self.encoder.num_layers, -1, self.encoder.hidden_dim * 2)

        hidden = self.h_t(hidden[-1, :])
        cell = self.c_t(cell[-1, :])

        input = trg[:, 0]

        if teacher_forcing_ratio == 1: # use teacher forcing
            teaching_trg = trg[:, :-1]
            padded_trg = pad_var_sequences(self.out_emb(teaching_trg), trg_len[:-1])
            padded_pred, hidden, cell = self.decoder(padded_trg, hidden, cell)
            pred = pack_var_sequences(padded_pred, total_length=trg_len_size)
            pred = self.o_head(pred)
            outputs[:, 1:, :] = pred
            return outputs

        random_teacher_forcing = torch.rand(trg_len_size - 1, device=self.device) < teacher_forcing_ratio
        for t in range(1, trg_len_size):
            input = self.out_emb(input)
            prediction, hidden, cell = self.decoder(input, hidden, cell)
            prediction = self.o_head(prediction)
            prob = self.softmax(prediction)
            outputs[:, t, :] = prediction
            top1 = prob.argmax(1)
            input = trg[:, t] if random_teacher_forcing[t-1] else top1

        return outputs
    
class SeqAttnLSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, bidirectional_encoder=True, num_layers=1, device="cuda"):
        super(SeqAttnLSTM, self).__init__()
        self.in_emb = nn.Embedding(vocab_size, input_size)
        self.out_emb = nn.Embedding(vocab_size, hidden_size)
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, bidirectional_encoder)
        self.decoder = LSTMCellDecoder(hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attn = AttnLayer()
        if bidirectional_encoder:
            t_hidden_size = 2 * hidden_size
        self.h_t = nn.Linear(t_hidden_size, hidden_size, bias=False)
        self.c_t = nn.Linear(t_hidden_size, hidden_size, bias=False)
        self.enc_t = nn.Linear(t_hidden_size, hidden_size, bias=False)
        self.o_head = nn.Linear(hidden_size * 2, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.device = device

    def forward(self, src, src_len, trg, trg_len, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len_size = trg.shape[1]
        trg_vocab_size = self.vocab_size

        outputs = torch.zeros(batch_size, trg_len_size, trg_vocab_size).to(self.device)

        src = self.in_emb(src)
        padded_src = pad_var_sequences(src, src_len)
        packed_src_out, hidden, cell = self.encoder(padded_src)
        src_out = pack_var_sequences(packed_src_out, padding_value=0., total_length=trg_len_size) # src_out (B, L, H * 2)
        src_out_t = self.enc_t(src_out) # src_out_t (B, L, H)
        src_mask = sequence_mask(src_len, max_length=trg_len_size, device=self.device) # src_mask (B, L)

        if self.encoder.bidirectional:
            hidden = hidden.view(self.encoder.num_layers, -1, self.encoder.hidden_dim * 2)
            cell = cell.view(self.encoder.num_layers, -1, self.encoder.hidden_dim * 2)

        hidden = self.h_t(hidden[-1, :])
        cell = self.c_t(cell[-1, :])

        input = trg[:, 0]

        if teacher_forcing_ratio == 1: # use teacher forcing
            teaching_trg = trg[:, :-1]
            padded_trg = pad_var_sequences(self.out_emb(teaching_trg), trg_len[:-1])
            padded_pred, hidden, cell = self.decoder(padded_trg, hidden, cell)
            pred = pack_var_sequences(padded_pred, total_length=trg_len_size)
            attn_pred, attn = self.attn(src_out_t, pred, src_mask)
            pred = torch.cat([pred, attn_pred], dim=2)
            pred = self.o_head(pred)
            outputs[:, 1:, :] = pred
            return outputs
        
        random_teacher_forcing = torch.rand(trg_len_size - 1, device=self.device) < teacher_forcing_ratio
        for t in range(1, trg_len_size):
            input = self.out_emb(input)
            prediction, hidden, cell = self.decoder(input, hidden, cell)
            attn_pred, attn = self.attn(src_out_t, prediction, src_mask)
            prediction = torch.cat([prediction, attn_pred.squeeze(1)], dim=1)
            prediction = self.o_head(prediction)
            prob = self.softmax(prediction)
            outputs[:, t, :] = prediction
            top1 = prob.argmax(1)
            input = trg[:, t] if random_teacher_forcing[t-1] else top1
        return outputs

