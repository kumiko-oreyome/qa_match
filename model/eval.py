import torch.nn.functional as F
import torch
import pandas as pd
from common.util import get_batch_of_device
from torch.utils.data import DataLoader
import numpy as np



def pairwise_match_question(question,answers,model,vocab,max_sentence_len,device,optional_fields=[],tokenizer=list,batch_size=32):
    from  common.datautil import  TextDataset
    n = len(answers)
    qds = TextDataset([question]*n,vocab,max_sentence_len,tokenizer)
    ans_ds = TextDataset(answers,vocab,max_sentence_len,tokenizer)
    q_dl = DataLoader(qds , batch_size=batch_size,shuffle=False)
    ans_dl = DataLoader(ans_ds , batch_size=batch_size,shuffle=False)
    sim_list = []
    for batch_q,batch_ans in zip(q_dl,ans_dl):
        batch_q =  get_batch_of_device(batch_q,device)
        batch_ans =  get_batch_of_device(batch_ans,device)
        qv = model.forward_question(batch_q)
        av = model.forward_answer(batch_ans )      
        qv = qv.detach()
        av = av.detach()
        sim = cosine_similarity(qv,av)
        sim_list.append(sim)
    sim_t = torch.cat(sim_list,0)
    _,sort_idx = torch.sort(sim_t,descending=True)
    if sim_t.is_cuda:
        sim_t = sim_t.cpu().numpy()
        sort_idx = sort_idx.cpu().numpy()
    if len(optional_fields) == 0 :
        return list(zip(np.array(answers)[sort_idx].tolist(),sim_t[ sort_idx].tolist()))
    else:
        return list(zip( *((np.array(answers)[sort_idx].tolist(),sim_t[sort_idx].tolist())+\
                     tuple([ np.array(field)[sort_idx].tolist() for field in optional_fields ]))))
                    

def rank_pred_by_sim(df):
    return   df.groupby(["question_id"]).apply(lambda x: x.sort_values(["sim"], ascending = False)).reset_index(drop=True)

def match_all(model,loader,device):
    qids = []
    ans_ids = []
    sim_list = []
    for batch in loader:
        batch =  get_batch_of_device(batch,device)
        qv = model.forward_question(batch['q'])
        av = model.forward_answer(batch['ans'])
        qv = qv.detach()
        av = av.detach()
        sim = cosine_similarity(qv,av)
        qids.append(batch['question_id'])
        ans_ids.append(batch['ans_id'])
        sim_list.append(sim)
    sim_t = torch.cat(sim_list,0)
    qid_t = torch.cat(qids,0)
    ans_t = torch.cat(ans_ids,0)

    if sim_t.is_cuda:
        d =  {'question_id':qid_t.cpu().numpy(),'ans_id':ans_t.cpu().numpy(),'sim':sim_t.cpu().numpy()}
    else:
        d = {'question_id':qid_t.numpy(),'ans_id':ans_t.numpy(),'sim':sim_t.numpy()}
    df = pd.DataFrame(data=d)
    df = rank_pred_by_sim(df)
    return df

class Evaluator():
    def __init__(self,evaluate_file):
        self.eva_df =  pd.read_csv(evaluate_file)
    #macro accuracy
    def evaluate_accuracy(self,preds,k=1):
        def accuracy_at_k(g):
            gdf = g.head(k)
            qid = g['question_id'].values[0]
            edf = self.eva_df.loc[(self.eva_df['question_id']==qid) & (self.eva_df['label']==1)]
            _ = set(gdf['ans_id'].values)&set(edf['ans_id'].values)
            #print(set(gdf['ans_id'].values))
            #print(set(edf['ans_id'].values))
            #print(len(_)/n)
            return len(_)/k
        accu = preds.groupby(['question_id']).apply(accuracy_at_k).reset_index(name='accu')    
        m = accu['accu'].mean()
        return m
    #macro hitrate
    def evaluate_hitrate(self,preds,k=1):
        def hit_at_k(g):
            gdf = g.head(k)
            qid = g['question_id'].values[0]
            edf = self.eva_df.loc[(self.eva_df['question_id']==qid) & (self.eva_df['label']==1)]
            _ = set(gdf['ans_id'].values)&set(edf['ans_id'].values)
            if len(_)>0:
                return 1.0
            return 0.0
        accu = preds.groupby(['question_id']).apply(hit_at_k).reset_index(name='accu')    
        m = accu['accu'].mean()
        return m


def cosine_similarity(q_vectors,a_vectors):
    a = torch.sum(q_vectors*a_vectors,dim=1)
    b = torch.norm(q_vectors,dim=1)*torch.norm(a_vectors,dim=1)
    return a/(b+1e-8)


def embedding_loss(pos_sims,neg_sims,M=0.2):
    diff = M-pos_sims+neg_sims
    _loss = torch.clamp(diff, min=0)
    return torch.mean(_loss)




#prediction: ranked answer id list , ground_truth : postive answer ids
def accuracy(prediction,ground_truth,k=1):
    pred_k = prediction[0:k]
    return len(set(pred_k)&set(ground_truth))/k


#print(cosine_similarity(torch.tensor([[1.0,2.0,2.0],[1.0,0.0,0.0]]),torch.tensor([[2.0,0.0,0.0],[5.0,0.0,0.0]])))