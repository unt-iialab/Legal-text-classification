#!/usr/bin/env python
# encoding: utf-8
'''
@author: haihua
@contact: haihua.chen@unt.edu
@file:domainextraction.py
@time:  7:19 PM
@desc:

'''

import os
import sys,re,collections,nltk
nltk.download('all')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize



pat_letter = re.compile(r'[^a-zA-Z \']+')
pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
pat_s = re.compile("(?<=[a-zA-Z])\'s")
pat_s2 = re.compile("(?<=s)\'s?")
pat_not = re.compile("(?<=[a-zA-Z])n\'t")
pat_would = re.compile("(?<=[a-zA-Z])\'d")
pat_will = re.compile("(?<=[a-zA-Z])\'ll")
pat_am = re.compile("(?<=[I|i])\'m")
pat_are = re.compile("(?<=[a-zA-Z])\'re")
pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

lmtzr = WordNetLemmatizer()


def replace_abbreviations(text):
    new_text = text
    new_text = pat_letter.sub(' ', text).strip().lower()
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text

# pos和tag有相似的地方，通过tag获得pos
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''

def merge(words):
    new_words = []
    for word in words:
        if word:
            tag = nltk.pos_tag(word_tokenize(word)) # tag is like [('bigger', 'JJR')]
            pos = get_wordnet_pos(tag[0][1])

            if pos:
                lemmatized_word = lmtzr.lemmatize(word, pos)
                new_words.append(lemmatized_word)
            else:
                new_words.append(word)
    return new_words

def get_words(file):
    with open (file) as f:
        words_box=[]
        for line in f:
            words_box.extend(merge(replace_abbreviations(line).split()))
    return collections.Counter(words_box)

def append_ext(words):
    new_words = []
    for item in words:
        word, count = item
        tag = nltk.pos_tag(word_tokenize(word))[0][1] # tag is like [('bigger', 'JJR')]
        new_words.append((word, count, tag))
    return new_words

if __name__ == '__main__':
    strPath = r'/media/isiia/TOSHIBA EXT/all_courtlistener_raw_text'
    floads = os.listdir(strPath)
    savePath = r'/media/isiia/TOSHIBA EXT/all_courtlistener_clean_text'
    for fload in sorted(floads):
        txts = os.listdir(strPath + "/" + fload)
        for txt in txts:
            txtPath = strPath + "/" + fload+ "/"+ txt
            words = get_words(txtPath)
            print(words)
            f_w = open(savePath+"/"+fload+"_"+txt, "w")
            f_w.write(' '.join(map(str,words)))
            f_w.close()
