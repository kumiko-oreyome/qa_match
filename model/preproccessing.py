from common.textutil import generate_ngram
from itertools import chain
def calculate_overlap_ngram_qa(q,a,n_grams):
    q = chain(*[generate_ngram(q,n) for n in n_grams ])
    a = chain(*[generate_ngram(a,n) for n in n_grams ])
    return set(q)&set(a)

