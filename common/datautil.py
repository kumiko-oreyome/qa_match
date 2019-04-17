import pandas as pd
from pandas import DataFrame 
import json
from torch.utils.data import Dataset,DataLoader
import numpy as np
from model.simplecnn import SimpleCNN
import torch
import random
import csv
from  model.eval import  accuracy,match_all
import  pickle as pkl







#def generate_evaluate_file(question_file,answer_file,answer_num_every_question):
#    question_df = pd.read_csv(question_file).set_index('question_id')
#    answer_df = pd.read_csv(answer_file).set_index('ans_id')   


def generate_fake_evaluate_file(question_file,answer_file,candidate_file,out_file,answer_num_every_question=10):
    #question_df = pd.read_csv(question_file).set_index('question_id')
    answer_df = pd.read_csv(answer_file).set_index('ans_id')
    candidate_df = pd.read_csv(candidate_file).set_index('question_id')
    trained_question_id = set(candidate_df.index.tolist())
    with open(out_file, 'w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["question_id","ans_id","label"]) #header
        for qid in trained_question_id:
            pos_ans_ids = answer_df.loc[answer_df['question_id']==qid].index.tolist()
            neg_ans_ids = answer_df.loc[answer_df['question_id']!=qid].index.tolist()
            neg_num = answer_num_every_question - len(pos_ans_ids)
            neg_ans_ids = random.choices(neg_ans_ids,k=neg_num)
            a = [(qid,pos_id,1) for pos_id in pos_ans_ids]
            b = [(qid,neg_id,0) for neg_id in neg_ans_ids]
            writer.writerows(a+b)

# EADGBE
# ADGCEA
# DAGCFAD   

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

    def to_tensor(self,vocab):
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
        return Text(s,self.max_sentence_len,self.tokenizer).to_tensor(self.vocab)

    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        q,a,label,question_id,ans_id = self._get_items_from_sample_row(row)
        return {'q':  self._text2array(q),'ans': self._text2array(a),'label':np.array(int(label),dtype=np.int32)\
                ,'question_id':np.array(int(question_id),dtype=np.int64),'ans_id':np.array(int(ans_id),dtype=np.int64)}


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
        return Text(s,self.max_sentence_len,self.tokenizer).to_tensor(self.vocab)
    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        q,pa,na = self._get_items_from_sample_row(row)
        return {'q':  self._text2array(q),'pos_ans': self._text2array(pa),'neg_ans':self._text2array(na)}
        
   
# generate_fake_evaluate_file('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_candidates.txt','./data/cMedQA2/small_fake_eval.csv')







#train_dataset = QAMatchDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_candidates.txt')
#dataloader = DataLoader(train_dataset, batch_size=32,shuffle=False)
#simplecnn = SimpleCNN(train_dataset.vocab,100,[(2,500),(3,500),(4,500)])
#print(len(train_dataset))
#for batch in dataloader:
#    o = simplecnn.forward_question(batch['q'])
#    print(o.size())

#vocab = train_dataset.vocab
#eval_dataset = QAEvaluateDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_fake_eval.csv',vocab)
#d = eval_dataset[0]
#print(vocab.decode(d['q']))
#print(vocab.decode(d['ans']))
#print(d['label'])
#print(d['question_id'])


#vocab = train_dataset.vocab
#eval_dataset = QAEvaluateDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_fake_eval.csv',vocab)
#dataloader = DataLoader(eval_dataset, batch_size=32,shuffle=False)
#for batch in dataloader:
#    print(batch['q'][0])
#    print(batch['label'])
#    print(batch['question_id'][0])
#    break


#dataloader = DataLoader(eval_dataset, batch_size=32,shuffle=False)
#for batch in dataloader:
#    print(batch[''])

#train_dataset = QAMatchDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_candidates.txt')
#
#
#d = train_dataset[0]
#print(train_dataset.vocab.decode(d['q']))
#print(train_dataset.vocab.decode(d['pos_ans']))
#print(train_dataset.vocab.decode(d['neg_ans']))
#d = train_dataset[1]
#print(train_dataset.vocab.decode(d['q']))
#print(train_dataset.vocab.decode(d['pos_ans']))
#print(train_dataset.vocab.decode(d['neg_ans']))
#
#
#


#def merge_cmed2_qa_files(question_file,answer_file):
#    question_df = pd.read_csv(question_file).set_index('question_id')
#    answer_df = pd.read_csv(answer_file).set_index('question_id')
#    joined_df = question_df.join(answer_df,lsuffix='_question',rsuffix='_answer')
#    return joined_df
#
#
#
#
#
#
#def generate_cmed2_examples(question_file,answer_file,sample_file,out_file):
#    question_df = pd.read_csv(question_file).set_index('question_id')
#    answer_df = pd.read_csv(answer_file).set_index('ans_id')
#    candidates_df = pd.read_csv(sample_file)
#    with open(out_file,'w',encoding='utf-8') as f:
#        for _,row in candidates_df.iterrows():
#            q = question_df.loc[row['question_id'],'content']
#            pa = answer_df.loc[row['pos_ans_id'],'content']
#            na = answer_df.loc[row['neg_ans_id'],'content']
#            s = json.dumps({'question':q,'pos_ans':pa,'neg_ans':na},ensure_ascii=False)
#            f.write(s+'\n')
#
#
#
#generate_cmed2_examples('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/train_candidates.txt','cemd2_train.jl')