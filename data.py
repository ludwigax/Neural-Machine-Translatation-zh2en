import os
import pickle as pkl

from typing import List, Tuple
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset

from build_data import build_data_tokenized

def pad_var_sequences(padded_sequences: torch.Tensor, lengths): # padded_sequences: (L, B, H)
    # note: the batch is a sorted list with descending length
    packed_sequences = rnn_utils.pack_padded_sequence(padded_sequences, lengths, batch_first=False)
    return packed_sequences

def sequence_collate_fn(batch: List[Tuple[int, int]], max_length: int = 100, device = "cpu"):
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    input_batch, target_batch = zip(*sorted_batch)
    lengths = [len(seq) for seq in input_batch]
    input_batch = rnn_utils.pad_sequence(input_batch, batch_first=False).to(device)
    target_batch = list(target_batch)
    for i in range(len(target_batch)):
        target_batch[i] = F.pad(target_batch[i], (0, max_length - len(target_batch[i])), value=0).to(device)
    target_batch = torch.stack(target_batch, dim=1) # (max_length, batch_size) batch_first = False
    return input_batch, target_batch, lengths


class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, max_length: int = 100, block_ids = 1):
        self.data_dir = data_dir
        self.block_ids = block_ids
        self.process(block_ids)

    def __len__(self):
        return len(self.en_sequences)

    def __getitem__(self, idx):
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
            