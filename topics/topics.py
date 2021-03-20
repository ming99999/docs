from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim import corpora, models
from gensim.models import Phrases

from collections import Counter

#vectorizing
def bow2vec(bow_list) :
    '''
    input : a list of bow lists
    output : vectors
    '''
    data_bow = []
    for item in bow_list :
        c = Counter()
        c.update(item)
        data_bow.append(c)
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(data_bow)

    return X

def bow2tfidfvec(bow_list) :
    '''
    input : a list of bow lists
    output : vectors
    '''
    data_bow = []
    for item in bow_list :
        data_bow.append(" ".join(item))
    tfidfvectorizer = TfidfVectorizer()
    X = tfidfvectorizer.fit_transform(data_bow)

    return X



