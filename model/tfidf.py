from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba as jb
from  model.eval import  Evaluator,rank_pred_by_sim
from common.datautil import QADataset
import pandas as pd
import pickle as pkl
from hanziconv import HanziConv
def read_txt_lines(path):
    lines = []
    with open(path,'r',encoding="utf-8") as f:
        for row in f:
            lines.append(row.rstrip('\n'))
    return lines

class Tokenizer():
    def __init__(self,stopword_path='stopwords.txt'):
        self.stopword_list = []
        if stopword_path is not None:
            self.stopword_list =  read_txt_lines(stopword_path)
    def tokenize(self,s):
        return [w for w in jb.cut(s, cut_all=False) if w not in  self.stopword_list] 


class Tfidf():
    def __init__(self,sentences,tokenize_func):
        self.sentences = sentences
        self.tokenize_func = tokenize_func
        print('make corpus')
        self.corpus =  self.make_corpus(self.sentences)
        print('corpus complete')
        #self.count_vectorizer = CountVectorizer(max_features=100000).fit(self.corpus)
        self.vectorizer = TfidfVectorizer(max_features=100000).fit(self.corpus)
    def make_corpus(self,sentences):
        wl =  list(map(lambda x:self.tokenize_func(x),sentences))
        return list(map(lambda x:" ".join(x),wl))
    #def count_transform(self,sentences):
    #    return self.count_vectorizer.transform(self.make_corpus(sentences)).toarray()
    def tfidf_transform(self,sentences):
        return  self.vectorizer.transform(self.make_corpus(sentences))
    def cosine_similarity(self,text1,text2):
        return  cosine_similarity(self.tfidf_transform([text1])[0],self.tfidf_transform([text2])[0])[0][0]

#class TfidfRanker():
#    def __init__(self,tfidf):
#        self.tfidf = tfidf




def test():
    tfidf = Tfidf(["我好你好大家好,高雄發大財","台灣阿誠世界偉人財神總統"],Tokenizer().tokenize)
    print(tfidf.tfidf_transform(["大家好 財神總統"]))
    print(tfidf.vectorizer.get_feature_names())


# build tfidf
#test()

BUILD_FLAG = False
PKL_FILE = 'tfidf.pkl'
if BUILD_FLAG:
    QUESTION_FILE = './data/cMedQA2/question.csv'
    ANSWR_FILE = './data/cMedQA2/answer.csv'
    qa_df = QADataset(QUESTION_FILE,ANSWR_FILE).get_df()
    questions = qa_df.loc[:,'question'].values
    answers = qa_df.loc[:,'answer'].values
    print('build tfidf vector')
    tfidf = Tfidf(questions+answers,Tokenizer().tokenize) 
    print('successfully build vector')
    with open('tfidf.pkl','wb') as f:
        pkl.dump(tfidf,f)
else:
    with open(PKL_FILE,'rb') as f:
        print('load tfidf vectorizer')
        tfidf = pkl.load(f)
        print('lennth %d'%(len(tfidf.vectorizer.get_feature_names())))

#EVAL_QUESTION_FILE = './data/app/food87_question_delete_no_answer.csv'
#EVAL_ANSER_FILE = './data/app/food87_answer_delete_no_answer.csv'
#EVAL_FILE = './data/app/food87_eval_delete_no_answer.csv'
#EVAL_QUESTION_FILE = './data/twe/twe_question.csv'
#EVAL_ANSER_FILE = './data/twe/twe_answer.csv'
#EVAL_FILE = './data/twe/twe_eval.csv'

EVAL_QUESTION_FILE = './data/cMedQA2/question.csv'
EVAL_ANSER_FILE = './data/cMedQA2/answer.csv'
EVAL_FILE = './data/cMedQA2/test_candidates.txt'

eval_q_df = pd.read_csv(EVAL_QUESTION_FILE).set_index('question_id')
eval_ans_df = pd.read_csv(EVAL_ANSER_FILE).set_index('ans_id')
eval_df =  pd.read_csv(EVAL_FILE)

#preds = pd.DataFrame(columns=['question_id','ans'])
d = {'question_id':[],'ans_id':[],'sim':[]}

for index,row in eval_df.iterrows():
    qid,ans_id = row['question_id'],row['ans_id']
    q = eval_q_df.loc[qid,'content']
    a = eval_ans_df.loc[ans_id,'content']
    d['question_id'].append(qid)
    d['ans_id'].append(ans_id)
    d['sim'].append(tfidf.cosine_similarity(HanziConv.toSimplified(q),HanziConv.toSimplified(a)))

pred = pd.DataFrame(data=d)
pred = rank_pred_by_sim(pred)
print(pred.head(200))

for k in range(5):
    print('hit rate @%d = %.3f'%(k+1,Evaluator(EVAL_FILE).evaluate_hitrate(pred,k+1)))





