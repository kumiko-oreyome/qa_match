import pandas as pd
from pandas import DataFrame 
import json
from torch.utils.data import Dataset,DataLoader
import numpy as np
from model.simplecnn import SimpleCNN
import torch
import random
from  model.eval import  accuracy,match_all
import  pickle as pkl


class DatasetMeta():
    def __init__(self,vocab,device=None,sentence_max_len=100,tokenizer=list):
        pass
        



def text2array(text,max_sentence_len,vocab,tokenizer):
    return Text(text,max_sentence_len,tokenizer).to_numpy(vocab)  



class DatasetMeta():
    def __init__(self):
        pass


class Text():
    def __init__(self,text,max_len,tokenizer=list):
        self.text = text
        self.max_len = max_len
        self.tokenizer = tokenizer

    def tokenize(self):
        return self.tokenizer(self.text)

    def numerize(self,vocab):
        tokens = self.tokenize()
        if len(tokens) > self.max_len:
           tokens = tokens[:self.max_len]
        ixs = vocab.encode(tokens)
        return ixs + [vocab._encode_one(vocab.PAD)] * (self.max_len-len(ixs))

    def to_numpy(self,vocab):
        return np.array(self.numerize(vocab),dtype=np.int64)

class Vocab():
    def __init__(self):
        self.ix2token = []
        self.token2ix = {}
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self._add_one_token(self.UNK)
        self._add_one_token(self.PAD)
    def _add_one_token(self,token):
        if token in self.token2ix:
            return
        ix = len(self.ix2token)
        self.token2ix[token] = ix
        self.ix2token.append(token)

    def size(self):
        return len(self.ix2token)

    def add_tokens(self,tokens):
        for token in tokens:
            self._add_one_token(token)

    def _encode_one(self,token):
        if token not in self.token2ix:
            return self.token2ix[self.UNK]
        return self.token2ix[token]

    def _decode_one(self,ix):
        if ix < 0 or ix >= len(self.ix2token):
            return self.UNK
        return self.ix2token[ix]
    
    def encode(self,tokens):
        return [ self._encode_one(token) for token in tokens]
    
    def decode(self,ixs,concat=True):
        tokens = [  self._decode_one(ix) for ix in ixs]
        if concat:
            return "".join(tokens)
        return tokens

    def save(self,path):
        with open(path,'wb') as f:
            pkl.dump(self,f)

    @classmethod
    def load(cls,path):
        with open(path,'rb') as f:
            return pkl.load(f)



class TextDataset(Dataset):
    def __init__(self,texts,vocab,max_sentence_len=100,tokenizer=list,device=None):
        self.texts = texts
        self.max_sentence_len = max_sentence_len
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
        self.max_sentence_len = max_sentence_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return text2array(self.texts[idx],self.max_sentence_len,self.vocab,self.tokenizer)
    

class QAEvaluateDataset(Dataset):
    def __init__(self,question_file,answer_file,eval_file,vocab,max_sentence_len=100,tokenizer=list,device=None):
        self.question_df = pd.read_csv(question_file).set_index('question_id')
        self.answer_df = pd.read_csv(answer_file).set_index('ans_id')
        self.sample_df = pd.read_csv(eval_file)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
        self.max_sentence_len = max_sentence_len

    def __len__(self):
        return len(self.sample_df)

    def _get_items_from_sample_row(self,row):
        q = self.question_df.loc[row['question_id'],'content']
        a = self.answer_df.loc[row['ans_id'],'content']
        return q,a,row['label'],row['question_id'],row['ans_id']

    def _text2array(self,s):
        return text2array(s,self.max_sentence_len,self.vocab,self.tokenizer)

    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        q,a,label,question_id,ans_id = self._get_items_from_sample_row(row)
        return {'q':  self._text2array(q),'ans': self._text2array(a),'label':np.array(int(label),dtype=np.int32)\
                ,'question_id':np.array(int(question_id),dtype=np.int64),'ans_id':np.array(int(ans_id),dtype=np.int64)}


class QADataset():
    def __init__(self,question_file,answer_file):
        self.question_df = pd.read_csv(question_file).rename({'content':'question'}, axis='columns')
        self.answer_df = pd.read_csv(answer_file).rename({'content':'answer'}, axis='columns')
        self._merge_question_answer_df()
    def _merge_question_answer_df(self):
        self.merged_df = self.question_df.set_index('question_id').join(self.answer_df.set_index('question_id'))
    def get_qa(self,question_id):
        q,a = self.answer_df.loc[question_id,'question'],self.answer_df.loc[question_id,'answer']
        return q,a
    def get_df(self):
        return  self.merged_df





class QAMatchDataset(Dataset):
    def __init__(self,question_file,answer_file,sample_file,vocab=None,max_sentence_len=100,tokenizer=list,device=None):
        self.question_df = pd.read_csv(question_file).set_index('question_id')
        self.answer_df = pd.read_csv(answer_file).set_index('ans_id')
        self.sample_df = pd.read_csv(sample_file)
        self.tokenizer = tokenizer
        self.device = device
        self.max_sentence_len = max_sentence_len
        if vocab is None:
            self._build_vocab()
        else:
            self.vocab = vocab

    def _build_vocab(self):
        vocab = Vocab()
        for _,row in self.sample_df.iterrows():
            q,pa,na = self._get_items_from_sample_row(row)
            vocab.add_tokens(self.tokenizer(q))
            vocab.add_tokens(self.tokenizer(pa))
            vocab.add_tokens(self.tokenizer(na))
        self.vocab = vocab
    def _get_items_from_sample_row(self,row):
        q = self.question_df.loc[row['question_id'],'content']
        pa = self.answer_df.loc[row['pos_ans_id'],'content']
        na = self.answer_df.loc[row['neg_ans_id'],'content']
        return q,pa,na
    def __len__(self):
        return len(self.sample_df)
    def _text2array(self,s):
        return text2array(s,self.max_sentence_len,self.vocab,self.tokenizer)
    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        q,pa,na = self._get_items_from_sample_row(row)
        return {'q':  self._text2array(q),'pos_ans': self._text2array(pa),'neg_ans':self._text2array(na)}
        

