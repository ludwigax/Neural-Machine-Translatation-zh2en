import numpy as np
import pickle
import time
import re

paralell_data = False

if paralell_data:
    punctuations = ['.', ',', ':', ';',]

    small_title_pattern = re.compile(r'(\b[A-Za-z0-9]+\.\s)|(\b[IVXLCDM]+\.\s)|(\([A-Za-z0-9]+\)\.\s)|(\[[A-Za-z0-9]+\]\.\s)')
    line_number_pattern = re.compile(r'^(?:\d+\.\d+|\d+\.|\([a-zA-Z]\))')

    sentences = []
    sentences_zh = []
    sentence_ids = []
    short_strings = []
    # small_title = []
    # phrases = []
    # rest_strings = []

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

    # for line in short_strings:
    #     words = line.split(' ')
    #     if small_title_pattern.match(line) is not None:
    #         line_type = 'small_title'
    #         small_title.append(line)
    #     elif len(words) <= 4:
    #         line_type = 'phrase'
    #         phrases.append(line)
    #     else:
    #         line_type = 'rest_string'
    #         rest_strings.append(line)


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
    # print('phrases: ', len(phrases), sum([len(s) for s in phrases]))
    # print('small_title: ', len(small_title), sum([len(s) for s in small_title]))
    # print('rest_strings: ', len(rest_strings), sum([len(s) for s in rest_strings]))


tokenized_data = False

## this is the code to tokenize the wiki_zh_2019 data, which is deprecated now.
# if tokenized_data:
#     with open('./wiki_zh_2019/AA/wiki_00', 'r', encoding='utf-8') as f:
#         lines = f.readlines()
        
#     line = lines[1].strip()

#     import json

#     line_dict = json.loads(line)
#     line_text = line_dict['text']
#     paragraphs = line_text.split('\n')
#     paragraphs = [p.strip() for p in paragraphs if p.strip() != '']

#     brackets_pattern = re.compile(r'(（[^)]*）)|(\([^)]*\))')
#     se_brackets_pattern = re.compile(r'(（[^）]*[^\u4e00-\u9fff]+[^）]*）)|(\([^)]*[^\u4e00-\u9fff]+[^)]*\))')
#     cites_pattern = re.compile(r'「([^」]*)」')

#     for i, p in enumerate(paragraphs):
#         p = se_brackets_pattern.sub('', p)
#         paragraphs[i] = cites_pattern.sub(r'\1', p)

#     print(line_dict.keys())
#     print(line_dict['title'])
#     print(paragraphs)



#     thu = thulac.thulac(seg_only=True)  

#     def build_vocab(texts):
#         vocab = {}
#         for text in texts:
#             words = thu.cut(text, text=True).split()
#             for word in words:
#                 if word not in vocab:
#                     vocab[word] = len(vocab)

#         control_symbols = ["[MASK]", "[SEP]", "[END]", "[CLS]"]
#         for symbol in control_symbols:
#             vocab[symbol] = len(vocab)
#         return vocab

#     def tokenize(text, vocab):
#         return [vocab[word] for word in thu.cut(text, text=True).split() if word in vocab]

#     texts = ''
#     vocab = build_vocab(texts)

#     # example of tokenizeed data by tokenizer
#     new_text = "这是另一个文本。"
#     tokenized_text = tokenize(new_text, vocab)

#     print("词表:", vocab)
#     print("分词结果:", tokenized_text)


bert_tokenized_data = True
if bert_tokenized_data:
    import berttokenizer as btkn
    import copy

    setting = copy.copy(btkn.bert_setting)
    setting['corpus_path'] = 'UNv1.0.en-zh.zh'
    setting['show_progress'] = True
    print(setting)

    bert_tkn = btkn.build_bert_tokenizer(setting=setting)
    



