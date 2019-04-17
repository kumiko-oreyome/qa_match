import torch.nn.functional as F
import torch

def cosine_similarity(q_vectors,a_vectors):
    return F.cosine_similarity(q_vectors,a_vectors,1)


def embedding_loss(pos_sims,neg_sims,M=0.2):
    diff = M-pos_sims-neg_sims
    _loss = torch.clamp(diff, min=0)
    return torch.mean(_loss)




def rank_by_similarity(items,sims):
    pass


def evalauate_accuracy_on_batch(batch_q,batch_ans,batch_label):
    pass
    



#prediction: ranked answer id list , ground_truth : postive answer ids
def accuracy(prediction,ground_truth,k=1):
    pred_k = prediction[0:k]
    return len(set(pred_k)&set(ground_truth))/k