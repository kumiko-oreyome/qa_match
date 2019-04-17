from torch.utils.data import Dataset,DataLoader
import numpy as np
from model.simplecnn import SimpleCNN
import torch
import random
import csv
from  model.eval import  accuracy,match_all,Evaluator
from common.datautil import QAMatchDataset,QAEvaluateDataset

train_dataset = QAMatchDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_candidates.txt')
dataloader = DataLoader(train_dataset, batch_size=32,shuffle=False)
simplecnn = SimpleCNN(train_dataset.vocab,100,[(2,500),(3,500),(4,500)])
vocab = train_dataset.vocab
eval_dataset = QAEvaluateDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_fake_eval.csv',vocab)
eval_loader = DataLoader(eval_dataset , batch_size=32,shuffle=False)

def test_match_all():
    d = match_all(simplecnn,eval_loader)
    print(d)

def test_evaulator():
    d = match_all(simplecnn,eval_loader)
    e = Evaluator('./data/cMedQA2/small_fake_eval.csv').evaluate_accuracy(d)



#test_match_all()
test_evaulator()



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