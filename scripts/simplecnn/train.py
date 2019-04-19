from argparse import ArgumentParser
from torch.utils.data import Dataset,DataLoader
from model.simplecnn import SimpleCNN
from  model.eval import  accuracy,match_all,Evaluator
from common.datautil import QAMatchDataset,QAEvaluateDataset
from model.learn import  MatchLearner
from torch.optim import SGD
from model.learn import Checkpoint
from common.util import  get_device
def parse_kernel_args(s):
    l = []
    for ss in s.split(","):
        ks,kn = ss.split(":")
        l.append((int(ks),int(kn)))
    return l


parser = ArgumentParser()
parser.add_argument('question_file')
parser.add_argument('answer_file')
parser.add_argument('train_file')
parser.add_argument('dev_file')
parser.add_argument('save_dir')

parser.add_argument('--emb-dim',default=100,type=int)
parser.add_argument('--kernels',default='2:100,3:100,4:100')

parser.add_argument('--lr',default=0.1,type=float)
parser.add_argument('--batch-size',default=32,type=int)
parser.add_argument('--epoch-num',default=100,type=int)

#parser.add_argument('--device',default=None)
args = parser.parse_args()

device = get_device()
train_dataset = QAMatchDataset(args.question_file,args.answer_file,args.train_file)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False)
simplecnn = SimpleCNN(train_dataset.vocab,args.emb_dim,parse_kernel_args(args.kernels)).to(device)
vocab = train_dataset.vocab
eval_dataset = QAEvaluateDataset(args.question_file,args.answer_file,args.dev_file,vocab)
eval_loader = DataLoader(eval_dataset , batch_size=args.batch_size,shuffle=False)
optimizer = SGD(simplecnn.parameters(), lr = args.lr, momentum=0.9)
leaner = MatchLearner(simplecnn,optimizer,device=device)
checkpoint = Checkpoint(args.save_dir,simplecnn,vocab)
eva = Evaluator(args.dev_file)
leaner.train(train_loader,eval_loader,checkpoint,eva,max_epoch=args.epoch_num,device=device)