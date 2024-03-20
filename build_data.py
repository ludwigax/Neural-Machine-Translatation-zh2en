import numpy as np
import pickle
import time
import tqdm
import re
import os

from berttokenizer import load_bert_tokenizer, encode

r"""
def build_paralell_data()
    open the UNv1.0.en-zh.en and UNv1.0.en-zh.zh files, which are raw txt files
    cut the sentences and short strings from the files
    save the sentences to sentences_en.pkl and sentences_zh.pkl
    sentence : list of str
"""

def build_paralell_data():
    punctuations = ['.', ',', ':', ';',]

    small_title_pattern = re.compile(r'(\b[A-Za-z0-9]+\.\s)|(\b[IVXLCDM]+\.\s)|(\([A-Za-z0-9]+\)\.\s)|(\[[A-Za-z0-9]+\]\.\s)')
    line_number_pattern = re.compile(r'^(?:\d+\.\d+|\d+\.|\([a-zA-Z]\))')

    sentences = []
    sentences_zh = []
    sentence_ids = []
    short_strings = []

    tic = time.time()
    with open('./UNv1.0.en-zh.en', 'r', encoding='utf-8') as f:
        i = 0
        line = f.readline().strip()
        while line != '':
            if line[-1] in punctuations:
                line_type = 'sentence'
                sentences.append(line)
                sentence_ids.append(i)
            else:
                line_type = 'short_string'
                # words = line.split(' ')
                short_strings.append(line)

            line = f.readline().strip()
            i += 1

    with open('./UNv1.0.en-zh.zh', 'r', encoding='utf-8') as f:
        temp_sentence = []
        line = f.readline().strip()
        while line != '':
            temp_sentence.append(line)
            line = f.readline().strip()

    sentences_zh = [temp_sentence[i] for i in sentence_ids]

    for i, line in enumerate(sentences):
        if line_number_pattern.search(line) is not None:
            sentences[i] = line_number_pattern.sub('', line).strip()

    for i, line in enumerate(sentences_zh):
        if line_number_pattern.search(line) is not None:
            sentences_zh[i] = line_number_pattern.sub('', line).strip()

    ids = np.arange(len(sentences))
    np.random.shuffle(ids)

    toc = time.time()
    print('time: ', toc - tic)
    print('sentences: ', len(sentences), sum([len(s) for s in sentences]))
    print('sentences_zh:', len(sentences_zh), sum([len(s) for s in sentences_zh]))

    with open('sentences_en.pkl', 'wb') as f:
        pickle.dump(sentences, f)
    with open('sentences_zh.pkl', 'wb') as f:
        pickle.dump(sentences_zh, f)
    print("[Done] Save the sentences to sentences_en.pkl and sentences_zh.pkl.")
    return


def build_data_tokenized(data_dir, lang='en', block_size=2000000):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # tokenized the data with no padding
    # the memory cost is high, so we save the tokenized data to files
    print(f"Processing {lang} data")
    ids_en, ids_zh = [], []
    if lang == 'en' or lang is None:
        with open('sentences_en.pkl', 'rb') as f:
            sentences_en = pickle.load(f)
        tkn_en = load_bert_tokenizer('bert_tokenizer_en.json')
        tkn_en.no_padding()

        for i, s in tqdm.tqdm(enumerate(sentences_en), desc='Tokenizing English'):
            ids = encode(tkn_en, s, rettype='ids')
            ids_en.append(ids)
            if (i + 1) % block_size == 0 or i == len(sentences_en) - 1:
                with open(os.path.join(data_dir, f'sentences_en_ids_{i//block_size}.pkl'), 'wb') as f:
                    pickle.dump(ids_en, f)
                ids_en = []
    
    if lang == 'zh' or lang is None:
        with open('sentences_zh.pkl', 'rb') as f:
            sentences_zh = pickle.load(f)
        tkn_zh = load_bert_tokenizer('bert_tokenizer_zh.json')
        tkn_zh.no_padding()
        for i, s in tqdm.tqdm(enumerate(sentences_zh), desc='Tokenizing Chinese'):
            ids = encode(tkn_zh, s, rettype='ids')
            ids_zh.append(ids)
            if (i + 1) % block_size == 0 or i == len(sentences_zh) - 1:
                with open(os.path.join(data_dir, f'sentences_zh_ids_{i//block_size}.pkl'), 'wb') as f:
                    pickle.dump(ids_zh, f)
                ids_zh = []
    print("Done")