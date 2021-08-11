# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import re
import itertools
import networkx
import collections
import math
import os.path
import sys

items = os.path.abspath(__file__).rsplit("/")
dir_path = '/'
for item in items:
    if item == 'Frism': break
    dir_path = os.path.join(dir_path, item)
sys.path.insert(1, dir_path)

from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from .config import config

eng_stop_words = stopwords.words('english')

def expand_keywords(keywords):
    size = len(keywords)
    result = []

    for i in range(size):
        for j in range(i+1, size+1):
            result.append(get_keyword(keywords, i, j))
    
    return result

def get_keyword(keywords, i, j):
    return " ".join(keywords[i:j])

def get_sentences_ko(text, keywords=[], remove_keywords=[]):
    candidates = sentence_prep(text)
    sentences = []
    index = 0

    for candidate in candidates:
        while len(candidate) and (candidate[-1] == '.' or candidate[-1] == ' '):
            candidate = candidate.strip(' ').strip('.')
        if len(candidate):
            sentences.append(Sentence(candidate + '.', index, \
                                                    keywords, remove_keywords))
            index += 1

    # remove tail max 5 times, remove keywords contain publisher and copyright
    for i in range(5):
        tail_remove_keyword_list = ignore_keyword_list + remove_keywords

        tail = sentences.pop()

        if any([x in tail.text for x in tail_remove_keyword_list if x]):
            print("remove tail [{}]".format(tail))
        else:
            sentences.append(tail)
            break

    return sentences

def get_sentences_en(text, keywords=[], remove_keywords=[]):
    sentences = []
    items = sent_tokenize(text)

    for i, item in enumerate(items):
        sentences.append(Sentence(item, i))    
    
    return sentences

def text_rank_token_extractor_ko(text, target_part=['Noun', 'Alpha']):
    return expand_pos_tagger(text, target_part) #TBD

def text_rank_token_extractor_en(text, target_part=['NN', 'NNS', 'NNP', 'NNPS']):
    noun_group = []
    noun_run = []
    ret = []

    for pos_tag_items in pos_tag(word_tokenize(text)):
        word, tag = pos_tag_items

        word = word.replace(".", "")

        if (word.startswith("$")) or (len(word) == 1) or \
                                        (word in eng_stop_words):
            tag = 'ignore'

        if tag in target_part:
            noun_run.append(word)        
        elif noun_run:
            noun_group.append(noun_run)
            noun_run = []
    else:
        if noun_run:
            noun_group.append(noun_run)
   
    for item in noun_group:
        ret += expand_keywords(item)
    
    return ret 

if config.LANG == 'ko':
    get_sentences = get_sentences_ko
    text_rank_token_extractor = text_rank_token_extractor_ko
elif config.LANG == 'en':
    get_sentences = get_sentences_en
    text_rank_token_extractor = text_rank_token_extractor_en


brackets = [ ("\(" , ")"), ("\[" , "]"), ("<" , ">"), ("\{" , "}") ]
doublespace_pattern = re.compile(u"\s+", re.UNICODE)
junk_filter = re.compile(u"^[a-zA-Z0-9?><;,.{}()[\]\-_+=!@#$%\^&*|/\n\t\s']{10,}", re.UNICODE)
head_filter = re.compile(u"[ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\.\?\!\s]+= ", re.UNICODE)

def clear_bracket(text):    
    for bracket in brackets :
        substr = []
        for head in re.finditer(bracket[0], text) :
            s = head.start()
            e = text[s:].find(bracket[1])
            substr.append(str(text[s:s+e+1]))
        for item in substr : text = text.replace(item, "")
    
    return text

bracket_pattern = re.compile(r"\([^()]*?\)|\[.*?\]|\<.*?\>|\{.*?\}|\【.*?】|\(.*?\(.*?\).*?\)")
def clear_bracket_regex(text):
    return bracket_pattern.sub("", text)

def xplit(value, delimiters):
    return re.split('|'.join([re.escape(delimiter) for delimiter in delimiters]), value)

def sentence_prep(text):
    cleared_sentences = []
    
    cleared = xplit(clear_bracket_regex(text), ['. ', '\n', '.\n'])
    
    for sentence in cleared :
        if sentence and not junk_filter.match(sentence) :
            sentence_h = head_filter.sub("", sentence.strip())
            if sentence_h :
                cleared_sentences.append(re.sub(r"[.]$", "" , doublespace_pattern.sub(" ", sentence_h.replace("&apos;", ""))))
    
    return cleared_sentences
   

ignore_keyword_list = ['기사제보 및 보도자료', '무단전재', '재배포']
    
def build_graph(sentences):
    graph = networkx.Graph()
    graph.add_nodes_from(sentences)
    pairs = list(itertools.combinations(sentences, 2))
    for pair in pairs:
        graph.add_edge(pair[0], pair[1], weight=co_occurence(pair[0], pair[1]))
    return graph

def build_keyword_graph(bow, co_occurance_count, emphasize_keywords, main_keywords):
    graph = networkx.Graph()
    graph.add_nodes_from(bow)
    pairs = list(itertools.combinations(bow, 2))
    for pair in pairs:
        key = "{}_{}".format(pair[0].lower(), pair[1].lower())
        count = co_occurance_count.get(key, 0)
        weight = keyword_relation_weight(bow[pair[0]],bow[pair[1]],count)

        if pair[0] in emphasize_keywords: weight = weight * 2
        if pair[0] in main_keywords: weight = weight * 3
        if pair[0] in config.FAIR_KEYWORD: weight = weight * 2
        weight = weight * additional_point_with_expand_noun(pair[0])

        graph.add_edge(pair[0], pair[1],weight=weight)

    return graph

def additional_point_with_expand_noun(keyword):
    size = len(keyword.split())

    weight_limit = 1.5
    
    additional_point = (math.log(size) / 2) + 1

    return min(additional_point, weight_limit)


def keyword_relation_weight(tf_a, tf_b, co_occur_count):
    return (2 * co_occur_count) / (tf_a + tf_b)

def co_occurence(sentence1, sentence2):
    p = sum((sentence1.bow & sentence2.bow).values())
    q = sum((sentence1.bow | sentence2.bow).values())
    return p / q if q else 0

def build_count_occurance(sentences):
    count_occurance = {}
    for sentence in sentences:
        pairs = list(itertools.combinations(sentence.nouns, 2))
        
        for pair in pairs:

            #if (' ' in pair[0] or ' ' in pair[1]) and \
            #   (pair[0] in pair[1] or pair[1] in pair[0]): continue

            key = "{}_{}".format(pair[0].lower(), pair[1].lower())
            count_occurance[key] = count_occurance.get(key, 0) + 1

            key = "{}_{}".format(pair[1].lower(), pair[0].lower())
            count_occurance[key] = count_occurance.get(key, 0) + 1

    return count_occurance


phrase = "(?:[가-힣| ]*?[가-힣]+)"
phrase_pattern = \
re.compile(r"[^가-힣]%c(%s)%c|[^가-힣]\"(%s)\"|[^가-힣]'(%s)'" % \
            (chr(8216), phrase, chr(8217), phrase, phrase))
def get_phrase_keywords(text):
    phrases = phrase_pattern.findall(text)

    phrase_keywords = []
    main_keywords = []

    for x in phrases:
        for y in x:
            if y: 
                items = y.split()

                if len(items) == 1:
                    main_keywords += items
                elif len(items) < 3:
                    phrase_keywords += items
    
    return phrase_keywords, main_keywords

def filter_keywords(keywords, remove_keywords):
    return list(set([x for x in keywords 
                    if (1 < len(x)) and (not (x in remove_keywords))]))

class Sentence:
    def __init__(self, text, index=0, keywords=[], remove_keywords=[]):
        self.index = index
        self.text = text
        self.nouns = text_rank_token_extractor(self.text)
        self.bow = collections.Counter(self.nouns)
        
        for keyword in keywords:
            if (keyword in self.text) and (not keyword in self.nouns) \
                                                and (1 < len(keyword)):
                self.nouns.append(keyword)
        
        for keyword in remove_keywords:
            if keyword in self.nouns:
                self.nouns.remove(keyword)

    def __unicode__(self):
        return self.text

    def __str__(self):
        return  "[{}] : {} \n {}".format(self.index, " ".join(self.nouns), self.text)

    def __repr__(self):
        try:
            return self.text.encode('utf-8')
        except:
            return self.text

    def __eq__(self, another):
        return hasattr(another, 'index') and self.index == another.index

    def __hash__(self):
        return self.index


class TextRank:
    def __init__(self, text, emphasize_sentance=None, \
                                    main_keywords=[], remove_keywords=[]):
        emphasize_keywords = [] 

        if emphasize_sentance:
            if "|" in emphasize_sentance:
                emphasize_sentance = emphasize_sentance.rsplit("|", 1)[0]
            emphasize_keywords += text_rank_token_extractor(emphasize_sentance)
        
        emphasize, main = get_phrase_keywords(text) 
        emphasize_keywords += emphasize
        main_keywords += main

        if emphasize_sentance:
            emphasize, main = get_phrase_keywords(emphasize_sentance) 
            emphasize_keywords += emphasize
            main_keywords += main
   
        # remove length 1 keyword
        emphasize_keywords = filter_keywords(emphasize_keywords, remove_keywords)
        main_keywords = filter_keywords(main_keywords, remove_keywords)

        frism.info("emphasize keywords [{}]".format(",".join(emphasize_keywords)))
        frism.info("main keywords [{}]".format(",".join(main_keywords)))
        frism.info("remove keywords [{}]".format(",".join(remove_keywords)))

        self.sentences = get_sentences(text, main_keywords, remove_keywords)
        self.graph = build_graph(self.sentences)
        self.pagerank = networkx.pagerank(self.graph, weight='weight')
        self.reordered = sorted(self.pagerank, key=self.pagerank.get, reverse=True)
        self.nouns = []
        for sentence in self.sentences:
            self.nouns += sentence.nouns
        self.bow = collections.Counter(self.nouns)

        # keyword pagerank make
        co_occurance_count = build_count_occurance(self.sentences)
        self.keyword_graph = build_keyword_graph(self.bow, co_occurance_count, emphasize_keywords, main_keywords)
        self.keyword_pagerank = networkx.pagerank(self.keyword_graph, weight='weight')
        self.keyword_reordered = sorted(self.keyword_pagerank, key=self.keyword_pagerank.get, reverse=True)

    def summarize(self, count=3):
        if not hasattr(self, 'reordered'):
            return ""
        candidates = self.reordered[:count]
        candidates = sorted(candidates, key=lambda sentence: sentence.index)
        return '\n'.join([candidate.text for candidate in candidates])

    def keywords(self, count=5):
        output = {}

        if hasattr(self, 'keyword_reordered'):
            for item in self.keyword_reordered[:count]:
                output[item] = self.keyword_pagerank[item]
        
        return output

if __name__ == '__main__':

   url = "https://www.reuters.com/article/us-europe-migrants-un/u-n-secretary-general-seeks-to-promote-global-migration-pact-amid-objections-idUSKBN1O90V5"
    url = "https://www.reuters.com/article/us-northkorea-southkorea-coal/south-korean-prosecutors-indict-four-for-importing-north-korean-coal-idUSKBN1O90TP"
    url = "https://www.reuters.com/article/uk-britain-eu-court/eu-court-ruling-boosts-brexit-opponents-idUSKBN1O90Q3"
    url = "https://www.reuters.com/article/us-biotech-hong-kong/hong-kong-market-could-open-cash-flood-gates-for-u-s-biotechs-idUSKBN1JA230"

    article = get_article(url)
    t = TextRank(article)
    print(t.summarize)
