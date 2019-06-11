from argparse import ArgumentParser
from model.learn import Checkpoint
from common.util import  get_device
from model.simplecnn import SimpleCNN
from model.eval import pairwise_match_question
import pandas as pd
import csv
# 將QA問答的輸出寫成CSV的形式(類似報告)方便分析
parser = ArgumentParser()
#parser.add_argument('question')
#parser.add_argument('answer_file')
#parser.add_argument('eval_file')
#parser.add_argument('ckpt_dir')
#parser.add_argument('--model-prefix',default='best')
#parser.add_argument('--batch-size',default=32)
#parser.add_argument('--device',default=None)
parser.add_argument('--simplified',default=True)
args = parser.parse_args()


questions = pd.read_csv('./data/app/food87_question_delete_no_answer.csv')['content'].values
answers = pd.read_csv('./data/app/food87_answer_delete_no_answer.csv')['content'].values
if args.simplified:
    from hanziconv import HanziConv
    questions = list(map(HanziConv.toSimplified,questions))
    answers = list(map(HanziConv.toSimplified,answers))


max_sentence_len = 100
device = get_device()
ckpt = Checkpoint('./checkpoints/default_simple_all')
ckpt.load('best', SimpleCNN)
vocab = ckpt.vocab
model = ckpt.model.to(device)


topk= 5
out_csv = 'report_fq.csv'
with open(out_csv, 'w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f,delimiter='\t')
    writer.writerow(["問題編號","問題","分數","答案"]) #header
    for  i,q in enumerate(questions) :
        results = pairwise_match_question(q,answers,model,vocab,max_sentence_len,device,batch_size=8)
        ra,rs = zip(*results[0:topk])
        writer.writerows([[i+1,HanziConv.toTraditional(q) , sim ,HanziConv.toTraditional(ans)] for ans,sim in zip(ra,rs)])
    