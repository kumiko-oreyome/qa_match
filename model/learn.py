
from eval import cosine_similarity,embedding_loss
import torch
import pandas as pd






#optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)


#class QAMatcher():
#    def __init__(self,model):
#        self.model = model
#        self.result = {}
#    def reset(self):
#        self.result = {}
#    
#    def add_match_result(self,sim,batch_qid,batch_ansid):


class MatchLearner():
    def __init__(self,model,optmizer,metric,validate_every,save_every):
        self.model = model
        self.optm = optmizer

        self.validate_every = validate_every
        self.save_every = save_every
        

    def validate(self,validate_loader): 
        for batch in validate_loader:
            qv = self.model.forward_question(batch['q'])
            av = self.model.forward_answer(batch['ans'])
            sim = cosine_similarity(qv,av)


    def train(self,train_loader,validate_loader,max_epoch):
        best_model = None
        best_accu = 0.0
        for epoch in range(max_epoch):
            loss_tot = 0.0
            print('epoch %d'%(epoch))
            for i,batch in enumerate(train_loader):
                qv = self.model.forward_question(batch['q'])
                pav = self.model.forward_answer(batch['pos_ans'])
                nav = self.model.forward_answer(batch['neg_ans'])
                sim_p = cosine_similarity(qv,pav)
                sim_n = cosine_similarity(qv,nav)
                loss = embedding_loss(sim_p,sim_n)
                loss_tot+=loss
                loss.backward()
                self.optm.step()

            if epoch % self.validate_every == 0:
                pass
            if epoch % self.save_every == 0 :
                pass





    def epoch_complete(self):
        pass

    def iteration_complete(self):
        pass


    
