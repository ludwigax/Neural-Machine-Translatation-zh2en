import os
import pickle as pkl

from typing import List, Tuple
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset

from build_data import build_data_tokenized

def pad_var_sequences(padded_sequences: torch.Tensor, lengths): # padded_sequences: (B, L, H)
    # note: the batch is a sorted list with descending length
    packed_sequences = rnn_utils.pack_padded_sequence(padded_sequences, lengths, batch_first=True)
    return packed_sequences

def pack_var_sequences(packed_sequences, padding_value = 0., total_length = None): # return : (B, L, Max_Len)
    padded_sequences, _ = rnn_utils.pad_packed_sequence(packed_sequences, batch_first=True, padding_value=padding_value, total_length=total_length)
    return padded_sequences

def sequence_mask(lengths, max_length = None, device = "cpu"): # mask 1 for padding, 0 for non-padding
    lengths = torch.tensor(lengths, dtype=torch.int64)
    if max_length is None:
        max_length = lengths.max()
    mask = torch.arange(max_length).expand(len(lengths), max_length) >= lengths.unsqueeze(1) # length is on cpu
    return mask.to(device)

def sequence_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], max_length: int = 100, device = "cpu"):
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    # padding src tensor
    src, trg = zip(*sorted_batch)
    src_len = [len(seq) for seq in src]
    src = rnn_utils.pad_sequence(src, batch_first=True).to(device)
    # padding trg tensor
    trg = list(trg)
    trg_len = [len(seq) for seq in trg]
    for i in range(len(trg)):
        trg[i] = F.pad(trg[i], (0, max_length - len(trg[i])), value=0).to(device)
    trg = torch.stack(trg, dim=0) # (batch_size, max_length) batch_first = True
    return (src, src_len), (trg, trg_len)


class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, max_length: int = 100, block_ids = 1):
        self.data_dir = data_dir
        self.block_ids = block_ids
        self.process(block_ids)

    def __len__(self):
        return len(self.en_sequences)

    def __getitem__(self, idx):
        if len(self.en_sequences[idx]) > 100:
            self.en_sequences[idx] = self.en_sequences[idx][:100]
        if len(self.zh_sequences[idx]) > 100:
            self.zh_sequences[idx] = self.zh_sequences[idx][:100]
        return torch.tensor(self.en_sequences[idx], dtype=torch.int64), torch.tensor(self.zh_sequences[idx], dtype=torch.int64)

    def process(self, block_ids = 1):
        self.block_ids = block_ids
        print(f"Load SequenceDataset at {self.data_dir}")
        if not os.path.exists(f'{self.data_dir}/sentences_en_ids_{block_ids}.pkl'):
            build_data_tokenized(self.data_dir, lang="en")
        if not os.path.exists(f'{self.data_dir}/sentences_zh_ids_{block_ids}.pkl'):
            build_data_tokenized(self.data_dir, lang="zh")

        print("Loading the data...")
        with open(f'{self.data_dir}/sentences_en_ids_{block_ids}.pkl', 'rb') as f:
            self.en_sequences = pkl.load(f)
        with open(f'{self.data_dir}/sentences_zh_ids_{block_ids}.pkl', 'rb') as f:
            self.zh_sequences = pkl.load(f)
        print("Done")
            