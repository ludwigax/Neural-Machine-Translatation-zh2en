from tokenizers import (
    Tokenizer, models, pre_tokenizers, decoders, processors, trainers
)

r"""
def build_bert_tokenizer(setting = {})
    using corpus_path as the training corpus

def load_bert_tokenizer(tokenizer_path = 'bert_tokenizer.json')
    load the tokenizer from the file

def encode(tokenizer: Tokenizer, sentence: str, rettype = 'ids')
    encode the sentence and return the ids or tokens

def encode_batch(tokenizer: Tokenizer, sentences: list)
    encode the sentence list and return the ids or tokens

def decode(tokenizer: Tokenizer, ids: list)
    decode the ids and return the sentence
"""


bert_setting = {
    'vocab_size': 20000,
    'min_frequency': 2,
    'show_progress': False,
    'corpus_path': 'corpus.txt',
    'tokenizer_path': 'bert_tokenizer.json'
}

def build_bert_tokenizer(setting = {}):
    bert_tokenizer = Tokenizer(models.WordPiece())
    bert_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    bert_tokenizer.decoder = decoders.WordPiece()
    bert_tokenizer.add_special_tokens( # the tokenizer intrinsicly add special tokens, this is just for demonstration
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    sep_ids = bert_tokenizer.token_to_id("[SEP]")
    cls_ids = bert_tokenizer.token_to_id("[CLS]")
    bert_tokenizer.post_processor = processors.BertProcessing(
        ("[SEP]", sep_ids),
        ("[CLS]", cls_ids),
    )
    bert_tokenizer.enable_padding(direction='right', pad_id=0, pad_token='[PAD]', length=100)

    trainer = trainers.WordPieceTrainer(
        vocab_size = setting.get('vocab_size', 20000),
        min_frequency = setting.get('min_frequency', 2),
        show_progress = setting.get('show_progress', False), 
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    bert_tokenizer.train(
        [setting.get('corpus_path', 'corpus.txt')], 
        trainer
    )

    save_path = setting.get('tokenizer_path', 'bert_tokenizer.json')
    bert_tokenizer.save(save_path, pretty=True)
    print(f"[Done] Save the tokenizer to {save_path}.")
    return bert_tokenizer

def load_bert_tokenizer(tokenizer_path = 'bert_tokenizer.json'):
    bert_tokenizer = Tokenizer.from_file(tokenizer_path)
    return bert_tokenizer

def encode(tokenizer: Tokenizer, sentence: str, rettype = 'ids'):
    if rettype == 'ids':
        return tokenizer.encode(sentence).ids
    elif rettype == 'tokens':
        return tokenizer.encode(sentence).tokens
    
def encode_batch(tokenizer: Tokenizer, sentences: list):
    return tokenizer.encode_batch(sentences, add_special_tokens=True)
    
def decode(tokenizer: Tokenizer, ids: list):
    return tokenizer.decode(ids, skip_special_tokens=True)