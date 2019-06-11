from argparse import ArgumentParser
from model.learn import Checkpoint
from common.util import  get_device
from model.simplecnn import SimpleCNN
from model.eval import pairwise_match_question
import pandas as pd

parser = ArgumentParser()
#parser.add_argument('question')
parser.add_argument('answer_file')
#parser.add_argument('eval_file')
#parser.add_argument('ckpt_dir')
#parser.add_argument('--model-prefix',default='best')
#parser.add_argument('--batch-size',default=32)
#parser.add_argument('--device',default=None)
parser.add_argument('--simplified',dest='simplified',action='store_true')
parser.add_argument('--need-url',action='store_true')
parser.set_defaults(simplified=True)
args = parser.parse_args()

df = pd.read_csv(args.answer_file)
answers = df['content'].values

if args.simplified:
    from hanziconv import HanziConv
    answers = list(map(HanziConv.toSimplified,answers))
if args.need_url:
    optional_fields = [df['url'].values]
else:
    optional_fields = []
#answers = ["可能是妇科炎症引起的建议放置达克宁栓口服三金片和龙胆泻肝丸如过不放心可以做一个支原体化验","这个是可以打的，注意饮食避免辛辣刺激性食物，注意多喝水，出水痘注意多喝水，不要吃肉的",\
#           "可能是会来，这种情况临床可以及时的妇科检查，需要清淡营养，少吃多餐易消化食物。黄体酮试试啊。","蔬菜水果不错","你的情况考虑局部有炎症的可能性大一些。建议服用红药片，乙酰螺旋霉素片等治疗观察。必要时拍X线明确诊断。"]

max_sentence_len = 100
device = get_device()
ckpt = Checkpoint('./checkpoints/default_simple_all')
ckpt.load('best', SimpleCNN)
vocab = ckpt.vocab
model = ckpt.model.to(device)

while True:
    question = input("Enter your question")
    if args.simplified:
        question = HanziConv.toSimplified(question)
    print('question is %s'%(question))
    print('model match')
    results = pairwise_match_question(question,answers,model,vocab,max_sentence_len,device,optional_fields,batch_size=8)
    print('results:')
    for r in results[0:5]:
        if not args.need_url:
            ans,sim = r
            print('%s --> %.3f'%(ans[0:100],sim))
        else:
            ans,sim,url = r
            print('%s --> %.3f  url:%s'%(ans[0:100],sim,url))        
#
#




