
from model.eval import cosine_similarity,embedding_loss,match_all
import torch
import pandas as pd
from common.datautil import Vocab
import os
from common.util import get_batch_of_device



class Checkpoint():
    def __init__(self,dirpath,model=None,vocab=None):
        self.dirpath = dirpath
        self.model = model
        self.vocab = vocab
        self.first_save = True
        self.vocab_path = '%s/vocab'%(self.dirpath)
        print('checkpont: %s'%(dirpath))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def save(self,prefix,**kwargs):
        if  self.first_save:
            self.first_save = True
            self.vocab.save(self.vocab_path)
        d = {
            'model': self.model.state_dict(),
            'model_hyper':  self.model.get_hypers()
            #'optimizer': self.opt.state_dict()
            }
        d.update(kwargs)
        torch.save(d, '%s/%s.ckpt'%(self.dirpath,prefix))

    def load(self,prefix,model_cls):
        self.vocab = Vocab.load(self.vocab_path)
        checkpoint = torch.load('%s/%s.ckpt'%(self.dirpath,prefix))
        self.model  = model_cls(vocab=self.vocab,**checkpoint['model_hyper']) 
        self.model.load_state_dict(checkpoint['model'])




    
class MatchLearner():
    def __init__(self,model,optmizer,device):
        self.model = model
        self.optm = optmizer
        self.device = device
    def train(self,train_loader,validate_loader,ckpt,evaluator,device,max_epoch=100,validate_every=5,save_every=5):
        best_accu = 0.0
        for epoch in range(max_epoch):
            loss_tot = 0.0
            cnt = 0
            print('epoch %d'%(epoch))
            for i,batch in enumerate(train_loader):
                if i % 5000 == 0:
                    print('iteration %d'%(i))
                self.optm.zero_grad()
                batch = get_batch_of_device(batch,self.device)
                qv = self.model.forward_question(batch['q'])
                pav = self.model.forward_answer(batch['pos_ans'])
                nav = self.model.forward_answer(batch['neg_ans'])
                sim_p = cosine_similarity(qv,pav)
                sim_n = cosine_similarity(qv,nav)
                loss = embedding_loss(sim_p,sim_n,weights=batch['sample_weight'].float())
                loss.backward()
                self.optm.step()
                loss_tot+=loss
                cnt+=1
            print('training loss %.3f'%(loss_tot/cnt))

            if epoch % validate_every == 0:
                print('validate')
                pred = match_all(self.model,validate_loader,device=self.device)
                accu = evaluator.evaluate_accuracy(pred)
                print('accuracy : %.3f'%(accu))
                if accu > best_accu:
                    best_accu = accu
                    if ckpt is not None:
                        ckpt.save('best',epoch=epoch)
                    

            if epoch % save_every == 0 and ckpt is not None:
               ckpt.save(str(epoch),epoch=epoch)