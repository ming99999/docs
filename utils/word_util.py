import re, os, math
import collections

from collections import Counter
from pymongo import MongoClient

doublespace_pattern = re.compile(u"\s+", re.UNICODE)
bracket_pattern = re.compile(r"\([^()]*?\)|\[.*?\]|\<.*?\>|\{.*?\}|\【.*?】|\(.*?\(.*?\).*?\)")
junk_filter = re.compile(u"^[a-zA-Z0-9?><;,.{}()[\]\-_+=!@#$%\^&*|/\n\t\s']{8,}", re.UNICODE)
head_filter = re.compile(u"[ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\.\?\!\s]+= ", re.UNICODE)
text_filter = re.compile(u'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\.\?\!-%]', re.UNICODE)
delete_filter = re.compile(r"[\'\"″“”‘’\?\!\.,…▲△□■※☆★▷▶▽▼『』#○●☆★＃◇◆ㅁ]")
space_filter = re.compile(r"[*·…‥♡♥\:=]|\.{2}")

#Basic

def only_text(text):
    return doublespace_pattern.sub(' ',text_filter.sub(' ', text)).strip()

def text_clearing(text):
    return doublespace_pattern.sub(' ', space_filter.sub(' ', delete_filter.sub('', text))).strip()

def clear_bracket_regex(text):
    return bracket_pattern.sub("", text)
        
def xplit(value, delimiters):
    return re.split('|'.join([re.escape(delimiter) for delimiter in delimiters]), value)


# Application
def sentence_prep(text):
    cleared_sentences = []
    cleared = xplit(clear_bracket_regex(text), ['. ', '\n', '.\n', '\r\n', '.\r\n'])
    
    for sentence in cleared :
        if sentence and not junk_filter.match(sentence) :
            sentence_h = head_filter.sub(" ", sentence.strip())
            if sentence_h :
                cleared_sentences.append(re.sub(r"[.]$", "" , doublespace_pattern.sub(" ", sentence_h.replace("&apos;", ""))))
    return cleared_sentences

def sentence_prep2(text):
    cleared_sentences = []
    cleared = xplit(clear_bracket_regex(text), ['. ', '\n', '.\n', '\r\n', '.\r\n'])
    
    for sentence in cleared :
        if sentence and not junk_filter.match(sentence) :
            cleared_sentences.append(re.sub(r"[.]$", "", text_clearing(sentence)))
    return cleared_sentences

def sentence_prep_str(text) :
    cleared_content = sentence_prep2(text)
    return "\n".join([x.strip() for x in cleared_content])

def tag_sellector(text, tagger, tag_list):
    return [x[0] for x in tagger.pos(text) if x[1] in tag_list]


def make_batch(list, batch_num = 5) :
    custom_batch = []
    period = int(len(list) / batch_num)

    for i in range(0, batch_num) :
        custom_batch.append(list[i * period : (i+1) * period])

    return custom_batch

# Trick
pos_set = ['Noun', 'Verb', 'Alpha']
'''
def selected_tokenizer(sent, pos = pos_set) :
    return [x[0] for x in _twitter.pos(sent) if len(x[0]) > 1 and x[1] in pos]
'''

# Sentence
def word_count(text, tagger) :
    
    tokenized = ["/".join(t) for t in tagger.pos(text, norm=True, stem=True)]
    word_counter = collections.Counter(tokenized)
    
    return word_counter

def cosine_similar(sent1, sent2, tokenizer = False) :
    vec1 = Counter()
    vec2 = Counter()

    if not tokenizer :
        tokenizer = lambda x : x.split()
    
    vec1.update(tokenizer(sent1))
    vec2.update(tokenizer(sent2))

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return round(float(numerator) / denominator, 3)


