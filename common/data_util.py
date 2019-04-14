import pandas as pd
from pandas import DataFrame 
import json
from torch.utils.data import Dataset,DataLoader
import numpy as np




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
            
    

def haskaki(batch):
    print(batch)
    print(batch[0])
    return []

class QAMatchDataset(Dataset):
    def __init__(self,question_file,answer_file,sample_file,mode,max_sentence_len=100,vocab=None,tokenizer=list,device=None):
        self.question_df = pd.read_csv(question_file).set_index('question_id')
        self.answer_df = pd.read_csv(answer_file).set_index('ans_id')
        self.sample_df = pd.read_csv(sample_file)
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_sentence_len = max_sentence_len
        if self.mode is not 'train' and vocab is None:
            assert False
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
        if self.mode == 'train':
            q = self.question_df.loc[row['question_id'],'content']
            pa = self.answer_df.loc[row['pos_ans_id'],'content']
            na = self.answer_df.loc[row['neg_ans_id'],'content']
            return q,pa,na
        else:
            assert False


    def __len__(self):
        return len(self.sample_df)
    

    def _text2arrat(self,s):
        return np.array(Text(s,self.max_sentence_len,self.tokenizer).numerize(self.vocab))

    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        q,pa,na = self._get_items_from_sample_row(row)
        return {'q':  self._text2arrat(q),'pos_ans': self._text2arrat(pa),'neg_pos':self._text2arrat(na)}
        
   

train_dataset = QAMatchDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_candidates.txt','train')
dataloader = DataLoader(train_dataset, batch_size=32,shuffle=False)
print(len(train_dataset))
for batch in dataloader:
    print(batch['q'][0])
    print(batch['pos_ans'][0])
    print(batch['neg_pos'][0])
    break




#train_dataset = QAMatchDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_candidates.txt','train')
#
#
#q,pa,na = train_dataset[0]
#print(train_dataset.vocab.decode(q))
#print(train_dataset.vocab.decode(pa))
#print(train_dataset.vocab.decode(na))
#q,pa,na = train_dataset[1]
#print(train_dataset.vocab.decode(q))
#print(train_dataset.vocab.decode(pa))
#print(train_dataset.vocab.decode(na))
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