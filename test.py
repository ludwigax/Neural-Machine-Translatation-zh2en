import berttokenizer as btkn
import pickle as pkl

from tqdm import tqdm

tkn = btkn.load_bert_tokenizer('bert_tokenizer_zh.json')
tkn.no_padding()

with open('sentences_zh.pkl', 'rb') as f:
    sentences_zh = pkl.load(f)

def calc_encode_length(tokenizer = None, tokenizer_path = "bert_tokenizer_{}.json", sentences = "sentences_{}.pkl", lang = "zh"):
    if not tokenizer:
        tokenizer = btkn.load_bert_tokenizer(tokenizer_path.format(lang))
    with open(sentences.format(lang), 'rb') as f:
        sentences = pkl.load(f)
    encode_length = []
    for se in tqdm(sentences):
        en = tkn.encode(se)
        encode_length.append(len(en.ids))
    return encode_length


r"""
this is a code to generate the token length distribution of the sentences in the dataset,
 and decide should the dataset be truncated or not.
"""
# encode_length = calc_encode_length(tkn, lang="zh")

# tkn = btkn.load_bert_tokenizer('bert_tokenizer_en.json')
# tkn.no_padding()

# encode_length_en = calc_encode_length(tkn, lang="en")

# import csv
# with open('encode_length.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['zh', 'en'])
#     for i in range(len(encode_length)):
#         writer.writerow([encode_length[i], encode_length_en[i]])


r"""
this is a code to plot the token length distribution of the sentences in the dataset.
"""
# import csv
# import matplotlib.pyplot as plt
# with open('encode_length.csv', 'r') as f:
#     reader = csv.reader(f)
#     next(reader)
#     encode_length = []
#     encode_length_en = []
#     for row in reader:
#         encode_length.append(int(row[0]))
#         encode_length_en.append(int(row[1]))

# # print(max(encode_length), max(encode_length_en))
# # enc_ratio = []
# # for i in range(len(encode_length)):
# #     enc_ratio.append(encode_length[i] / encode_length_en[i])

# plt.hist(encode_length, bins=100, range=(0, 500), alpha=0.5, label='zh')
# plt.hist(encode_length_en, bins=100, range=(0, 500), alpha=0.5, label='en')
# # plt.hist(enc_ratio, bins=100, range=(0, 10), alpha=0.5, label='zh/en')
# plt.legend(loc='upper right')
# plt.show()