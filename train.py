from models import (
    SeqAttnLSTM,
    SeqLSTMv1,
    SeqLSTMv2,
)
from data import SequenceDataset, sequence_collate_fn

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import numpy as np
import wandb

import berttokenizer as btkn
from nltk.translate.bleu_score import sentence_bleu
perplexity = lambda x: torch.exp(x)

def train(model: nn.Module, dataloader: DataLoader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src_block, trg_block) in tqdm.tqdm(enumerate(dataloader)):
        # the src and trg are already on the device
        optimizer.zero_grad()
        src, src_len = src_block
        trg, trg_len = trg_block

        output = model(src, src_len, trg, trg_len, teacher_forcing_ratio=0.5)
        norm = torch.norm(output, p=2).detach()

        B = output.shape[0]
        P = output.shape[2]
        output = output[:, 1:, :].reshape(-1, P)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        # wandb.log({"loss": loss.item(), "norm": norm.item()})
        print(f"Batch {i} | Loss: {loss.item():.3f} | Perplexity: {perplexity(loss.item()):.3f} | Norm: {norm:.3f}")
    return epoch_loss / len(dataloader)

def test(model: nn.Module, dataloader: DataLoader):
    model.eval()
    tkn_en = btkn.load_bert_tokenizer("bert_tokenizer_en.json")

    for i, (src_block, trg_block) in enumerate(dataloader):
        src, src_len = src_block
        trg, trg_len = trg_block

        output, preds = model.predict(src, src_len, max_len=len(trg_len))
        B = output.shape[0]
        P = output.shape[2]
        output = output[:, 1:, :].reshape(-1, P)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        print(f"Batch {i} | Perplexity: {perplexity(loss.item()):.3f}")
        
        if B == 1:
            sentence = tkn_en.decode(preds)
        else:
            sentence = []
            for j in range(B):
                sentence.append(
                    tkn_en.decode(preds[j])
                )
        # wandb.log({"test_loss": loss.item()}


if __name__=="__main__":
    input_dim = 256
    hidden_dim = 256
    bidirectional = True
    num_layers = 2
    vocab_size = 20000

    epochs = 25
    clip = 1.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_idx = 0

    # wandb.init(project="seq2seq", name="lstm_test_version_1")

    seq2seq = SeqAttnLSTM(vocab_size, input_dim, hidden_dim, bidirectional, num_layers, device=device).to(device)

    optimizer = optim.Adam(seq2seq.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    dataset = SequenceDataset("./sequencedataset", max_length=100, block_ids=1)

    def collate_fn(batch):
        return sequence_collate_fn(batch, 100, device=device) # set device in collate_fn

    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    trn_idx = idx[:int(len(dataset) * 0.8)]
    val_idx = idx[int(len(dataset) * 0.8):]

    train_loader = DataLoader(Subset(dataset, trn_idx), batch_size=100, shuffle=True, collate_fn=collate_fn)

    print(seq2seq)

    for e in range(epochs):
        
        train_loss = train(seq2seq, train_loader, optimizer, criterion, clip)
        # wandb.log({"epoch": e, "train_loss": train_loss})
        print(f'Epoch: {e+1:02} | Train Loss: {train_loss:.3f}')

        if (e + 1) % 5 == 0:
            block_ids = dataset.block_ids
            if block_ids == 5:
                block_ids = -1
            dataset.process(block_ids + 1)
            train_loader = DataLoader(Subset(dataset, trn_idx), batch_size=100, shuffle=True, collate_fn=collate_fn)

    # wandb.finish()