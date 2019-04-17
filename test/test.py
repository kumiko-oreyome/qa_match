from torch.utils.data import Dataset,DataLoader
import numpy as np
from model.simplecnn import SimpleCNN
import torch
import random
import csv
from  model.eval import  accuracy,match_all,Evaluator
from common.datautil import QAMatchDataset,QAEvaluateDataset
from model.learn import  MatchLearner
from torch.optim import SGD
from model.learn import Checkpoint

train_dataset = QAMatchDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_candidates.txt')
train_loader = DataLoader(train_dataset, batch_size=32,shuffle=False)
simplecnn = SimpleCNN(train_dataset.vocab,100,[(2,10),(3,10),(4,10)])
vocab = train_dataset.vocab
eval_dataset = QAEvaluateDataset('./data/cMedQA2/question.csv','./data/cMedQA2/answer.csv','./data/cMedQA2/small_fake_eval.csv',vocab)
eval_loader = DataLoader(eval_dataset , batch_size=32,shuffle=False)
optimizer = SGD(simplecnn.parameters(), lr = 0.01, momentum=0.9)

def test_match_all():
    d = match_all(simplecnn,eval_loader)
    print(d)

def test_evaulator():
    d = match_all(simplecnn,eval_loader)
    e = Evaluator('./data/cMedQA2/small_fake_eval.csv').evaluate_accuracy(d)

def test_train():
    leaner = MatchLearner(simplecnn,optimizer)
    checkpoint = Checkpoint('./checkpoints/small',simplecnn,vocab)
    eva = Evaluator('./data/cMedQA2/small_fake_eval.csv')
    leaner.train(train_loader,eval_loader,checkpoint,eva)

def test_reload():
    checkpoint = Checkpoint('./checkpoints/small')
    checkpoint.load('best', SimpleCNN)
    

test_reload()
#test_train()
#test_match_all()
#test_evaulator()



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