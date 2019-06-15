import jieba as jb
import json
from hanziconv import HanziConv
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer






def to_simplified_sentences(sentences):
    return [HanziConv.toSimplified(s) for s in sentences]

def read_txt_lines(path):
    lines = []
    with open(path,'r',encoding="utf-8") as f:
        for row in f:
            lines.append(row.rstrip('\n'))
    return lines
    
def read_json_utf8(path):
    with open(path,'r',encoding="utf-8") as f:
        obj = json.load(f)
    return obj

def write_json_utf8(path,obj):
    with open(path,'w',encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False)


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
        self.corpus =  self.make_corpus(self.sentences)
        self.count_vectorizer = CountVectorizer().fit(self.corpus)
        self.tfidf_transformer = TfidfTransformer().fit(self.count_transform(sentences))
    def make_corpus(self,sentences):
        return list(map(lambda x:self.tokenize_func(x),sentences))
    def count_transform(self,sentences):
        return self.count_vectorizer.transform(self.make_corpus(sentences)).toarray()
    def tfidf_transform(self,sentences):
        return self.tfidf_transformer.transform(self.count_transform(sentences))