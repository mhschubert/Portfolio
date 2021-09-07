import copy
import os
import sys

import torch
import dill as pickle
import srsly
import spacy
import pandas as pd
from torchtext.legacy import data, datasets

import re


def fix_spacy(nlp):
    #exchange patterns to assess contractions correctly
    patterns = srsly.read_json("../data/spacy_files/ar_patterns.json")
    old = nlp.remove_pipe("attribute_ruler")
    del old
    ar = nlp.add_pipe("attribute_ruler", before="lemmatizer")
    ar.add_patterns(patterns)

    return nlp

class Tokenizer(object):

    def __init__(self, lang:str):
        self.lang_dic = {'deu': 'de_dep_news_trf',
                         'eng' : 'en_core_web_trf'}
        if lang in self.lang_dic.keys():
            self.nlp = spacy.load(self.lang_dic[lang])
            if lang == 'eng':
                self.nlp = fix_spacy(self.nlp)
            self.tokenizer = self.nlp.tokenizer
            del self.nlp
        else:
            sys.exit('Error - language not available. Please download and add to spacy dic')

    def add_language(self, lang_code:str, spacy_name:str):
        self.lang_dic[lang_code] = spacy_name
        print('added language')


    def tokenize(self, text):

        text = re.sub( r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(text))
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"\.+", ".", text)
        text = re.sub(r"\,+", ",", text)
        #text = text.lower()
        text = [tkn.text for tkn in self.tokenizer(text) if tkn.text != " "]
        return text

def create_data_fields(tokenizers:[Tokenizer, Tokenizer]):

    src = data.Field(lower=False, tokenize=tokenizers[0].tokenize, batch_first=True, pad_token='<PAD>')
    trgt = data.Field(lower=False, tokenize=tokenizers[1].tokenize, init_token='<SoS>', eos_token='<EoS>',
                      pad_token='<PAD>', batch_first=True)

    return [('src', src), ('trgt', trgt)]

def create_dataset(path:str, lang_from_to:str, maxlen:int, batchsize:int,
                   device:torch.device, train=True, tokenizers=None, max_vocab_in=None, max_vocab_out=None):

    load_str = 'dev' if train else 'test'
    spltr = lang_from_to.split(' ')
    src = spltr[0]
    trgt = spltr[1]

    df = None
    for file in os.listdir(path):
        if load_str in file and os.path.isfile(os.path.join(path, file)):
            if df is None:
                df = pd.read_csv(os.path.join(path, file), sep=r'\t', header=None)
            else:
                df = pd.concat([df, pd.read_csv(os.path.join(path, file), sep=r'\t', header=None)])
    df.reset_index(inplace=True, drop=True)
    #fix src-target order
    cols = ['src_lang', 'target_lang', 'src', 'trgt']
    if df.columns.values[0] != src:
        cols = ['target_lang', 'src_lang', 'trgt', 'src']
    df.columns = cols
    df.drop(labels=['src_lang', 'target_lang'], axis=1, inplace=True)
    #create len mask
    if maxlen is not None:
        mask = (df['src'].apply(lambda x: len(x.split(' ')) <= maxlen)) & \
               (df['trgt'].apply(lambda x: len(x.split(' ')) <= maxlen))
        mask = list(mask)
        df = copy.deepcopy(df.loc[mask,:])
    leng = df.shape[0]
    os.makedirs(os.path.join(path, 'tmp'), exist_ok=True)
    df.to_csv(os.path.join(path, 'tmp', 'tmp_train_{}_{}_{}.csv'.format(train,src, trgt)), index=False)
    del df
    print('making tokenizer...')
    if tokenizers is None:
        tokenizers = []
        tokenizers.append(Tokenizer(src))
        tokenizers.append(Tokenizer(trgt))
    if train:
        fields = create_data_fields(tokenizers)
    else:
        with open(os.path.join(path, '../..', 'models/weights/vocab/src_{}_vocab.p'.format(src)), 'rb') as r:
            srcf = pickle.load(r)
        with open(os.path.join(path, '../..', 'models/weights/vocab/trgt_{}_vocab.p'.format(trgt)), 'rb') as r:
            trgtf = pickle.load(r)
        fields = [('src',srcf),('trgt', trgtf)]

    print('make data loader and batcher...')
    dat = data.TabularDataset(os.path.join(path,'tmp' ,'tmp_train_{}_{}_{}.csv'.format(train,src, trgt)), format='csv',
                              fields=fields)

    if not train:
        batchsize = leng
    dat_iter = data.BucketIterator(dat, batch_size=batchsize, train=train, shuffle=train,
                  device = device, sort_key=lambda x: (len(x.src), len(x.trgt)))
    if train:
        fields[0][1].build_vocab(dat, max_size=max_vocab_in)
        fields[1][1].build_vocab(dat, max_size=max_vocab_out)
        os.makedirs(os.path.join(path, '../..', 'models/weights/vocab/'), exist_ok=True)
        with open(os.path.join(path, '../..', 'models/weights/vocab/src_{}_vocab.p'.format(src)), 'wb') as w:
            pickle.dump(fields[0][1], w)
        with open(os.path.join(path, '../..', 'models/weights/vocab/trgt_{}_vocab.p'.format(trgt)), 'wb') as w:
            pickle.dump(fields[1][1], w)

    if train:
        return dat_iter, leng, fields[0][1].vocab.stoi['<PAD>'], fields[1][1].vocab.stoi['<PAD>']
    else:
        src_vocab = fields[0][1].vocab
        trgt_vocab = fields[1][1].vocab
        return dat_iter, leng, fields[0][1].vocab.stoi['<PAD>'], \
               fields[1][1].vocab.stoi['<PAD>'], src_vocab, trgt_vocab

