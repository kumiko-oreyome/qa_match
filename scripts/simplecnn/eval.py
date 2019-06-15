
from model.learn import Checkpoint
from model.simplecnn import SimpleCNN
from  model.eval import  match_all,Evaluator
from common.datautil import QAEvaluateDataset
from argparse import ArgumentParser
from torch.utils.data import Dataset,DataLoader
from common.datautil import QAMatchDataset,QAEvaluateDataset
from common.util import  get_device

parser = ArgumentParser()
parser.add_argument('question_file')
parser.add_argument('answer_file')
parser.add_argument('eval_file')
parser.add_argument('ckpt_dir')

parser.add_argument('--model-prefix',default='best')

parser.add_argument('--batch-size',default=32)
#parser.add_argument('--device',default=None)
args = parser.parse_args()

device = get_device()
checkpoint = Checkpoint(args.ckpt_dir)
checkpoint.load(args.model_prefix, SimpleCNN)
simplecnn = checkpoint.model.to(device)
vocab = checkpoint.vocab


eval_dataset = QAEvaluateDataset(args.question_file,args.answer_file,args.eval_file,vocab)
eval_loader = DataLoader(eval_dataset , batch_size=args.batch_size,shuffle=False)

d = match_all(simplecnn,eval_loader,device)
for k in range(5):
    #accu = Evaluator(args.eval_file).evaluate_accuracy(d,k+1)
    accu = Evaluator(args.eval_file).evaluate_hitrate(d,k+1)
    print('accuracy %d of %s is %.3f'%(k+1,args.eval_file,accu))
