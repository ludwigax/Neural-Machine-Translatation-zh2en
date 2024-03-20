import berttokenizer as btkn

import pickle as pkl
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from models import build_bert_model
from tqdm import tqdm

import numpy as np

def create_tokenizer(lang='en'):
    import copy

    setting = copy.copy(btkn.bert_setting)
    setting['corpus_path'] = 'UNv1.0.en-zh.zh'
    setting['show_progress'] = True
    setting['tokenizer_path'] = 'bert_tokenizer_zh.json'
    print(setting)

    bert_tkn = btkn.build_bert_tokenizer(setting=setting)


class ParallelDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizers: dict, dataset_path: str, max_length=100, split_blocks=True, block: int | None = 0):
        self.source_path = "{}_zh.pkl".format(dataset_path)
        self.target_path = "{}_en.pkl".format(dataset_path)
        self.tokenizer_zh = tokenizers['zh']
        self.tokenizer_zh.enable_truncation(max_length)
        self.tokenizer_en = tokenizers['en']
        self.tokenizer_en.enable_truncation(max_length)
        self.max_length = max_length
        self.split_blocks = split_blocks # the data is always large enough to be split into blocks
        self.block = block

        if True:
            self.process()

    def __len__(self):
        return len(self._data['source']['ids'])

    def __getitem__(self, idx):
        input_ids = torch.tensor(self._data['source']['ids'][idx]).squeeze()
        attention_mask = torch.tensor(self._data['source']['attention_mask'][idx]).squeeze()
        labels = torch.tensor(self._data['target']['ids'][idx]).squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": labels,
        }
    
    def _init_data(self):
        self._data = {}
        self._data['source'] = {
            'ids': [],
            'attention_mask': [],
        }
        self._data['target'] = {
            'ids': [],
            'attention_mask': [],
        }

    def _save_data(self, block):
        with open(f'encoded_data_{block}.pkl', 'wb') as f:
            pkl.dump(self._data, f)
    
    def process(self):
        import os
        results = []
        if self.block is not None and os.path.exists(f'encoded_data_{self.block}.pkl'):
            with open(f'encoded_data_{self.block}.pkl', 'rb') as f:
                self._data = pkl.load(f)
            return
        
        with open(self.source_path, 'rb') as f:
            self.source_texts = pkl.load(f)
        with open(self.target_path, 'rb') as f:
            self.target_texts = pkl.load(f)

        idx = np.arange(len(self.source_texts))
        np.random.shuffle(idx)
        
        print('Encoding...')
        self._init_data()
        for i in tqdm(range(len(self.source_texts)//1000 + 1)):
            if (i + 1) % 2000 == 0:
                self._save_data(i//2000)
                self._init_data()

            sdx = i * 1000
            edx = (i + 1) * 1000
            if edx > len(self.source_texts):
                edx = len(self.source_texts)
            
            enc_zh = btkn.encode_batch(self.tokenizer_zh, [self.source_texts[i] for i in idx[sdx:edx]])
            enc_en = btkn.encode_batch(self.tokenizer_en, [self.target_texts[i] for i in idx[sdx:edx]])
            self._data['source']['ids'] += [en.ids for en in enc_zh]
            self._data['source']['attention_mask'] += [en.attention_mask for en in enc_zh]
            self._data['target']['ids'] += [en.ids for en in enc_en]
            self._data['target']['attention_mask'] += [en.attention_mask for en in enc_en]
        self._save_data(i//2000)
        

def collate_fn(batch):
    batch = default_collate(batch)
    
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "label_ids": batch["label_ids"],
    }


def compute_loss(model, inputs, return_outputs=False):
    tokenizer_en = model.tokenizer_en
    outputs = model(**inputs)
    logits = outputs

    labels = inputs["label_ids"]
    logits = logits[:, :, :tokenizer_en.get_vocab_size()]

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_en.token_to_id("[PAD]"))
    loss = loss_fct(logits.view(-1, tokenizer_en.get_vocab_size()), labels.view(-1))

    return (loss, outputs) if return_outputs else loss

class ParallelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return compute_loss(model, inputs, return_outputs=return_outputs)


if __name__ == '__main__':
    tokenizer_en = btkn.load_bert_tokenizer('bert_tokenizer_en.json')
    tokenizer_zh = btkn.load_bert_tokenizer('bert_tokenizer_zh.json')
    tokenizers = {
        'en': tokenizer_en,
        'zh': tokenizer_zh,
    }

    model = build_bert_model(
        tokenizers = tokenizers
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate = 1e-4,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_dir="./logs",
        save_strategy="epoch",
        logging_strategy="epoch",
    )

    dataset = ParallelDataset(tokenizers, 'sentences', max_length=100, block=0)

    trainer = ParallelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )


    print('Start training...')
    trainer.train()

    # model.to('cuda')
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    # for batch in dataloader:
    #     batch = {k: v.to('cuda') for k, v in batch.items()}
    #     loss = compute_loss(model, batch, return_outputs=False)
    #     print(loss)
    #     break
