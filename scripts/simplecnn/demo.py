from argparse import ArgumentParser
from model.learn import Checkpoint
from common.util import  get_device
from model.simplecnn import SimpleCNN
from model.eval import pairwise_match_question
#parser = ArgumentParser()

question = "腰酸并伴有小腹胀痛怎么回事啊，而且引导有少量的血？"
answers = ["可能是妇科炎症引起的建议放置达克宁栓口服三金片和龙胆泻肝丸如过不放心可以做一个支原体化验","这个是可以打的，注意饮食避免辛辣刺激性食物，注意多喝水，出水痘注意多喝水，不要吃肉的",\
           "可能是会来，这种情况临床可以及时的妇科检查，需要清淡营养，少吃多餐易消化食物。黄体酮试试啊。","12345",""]

max_sentence_len = 100
device = get_device()

ckpt = Checkpoint('./checkpoints/simplecnn_default')
ckpt.load('best', SimpleCNN)
vocab = ckpt.vocab
model = ckpt.model.to(device)


print('question is %s'%(question))
print('model match')
results = pairwise_match_question(question,answers,model,vocab,max_sentence_len,device,batch_size=2)
print('results:')
for ans,sim in results:
    print('%s --> %.3f'%(ans[0:100],sim))






