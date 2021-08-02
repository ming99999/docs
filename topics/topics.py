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

def get_gensim_tfidf(token_list, min_count = 3) :
    '''
    input : a list of token lists
    output : gensim.corpora.Dictionary, a list of TfIdfvectors
    '''
    word_counter = Counter()
    for tokens in token_list :
        word_counter.update(tokens)
    dictionary_ko = corpora.Dictionary(token_list)
    removal_word_idxs = {
            dictionary_ko.token2id[word] for word, count in word_counter.items()
            if count < min_count
            }
    dictionary_ko.filter_tokens(removal_word_idxs)
    dictionary_ko.compactify()

    tf_ko = [(idx, word_counter.get(word, 0)) for word, idx in dictionary_ko.toekn2id.items()]
    tfidf_model_ko = models.TfidfModel(tf_ko)
    tfidf_ko = tfidf_model_ko[tf_ko]

    return dictionary_ko, tfidf_ko
