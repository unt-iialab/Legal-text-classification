#!/usr/bin/env python
# encoding: utf-8
'''
@author: haihua
@contact: haihua.chen@unt.edu
@file:dpdcextraction.py
@time:  8:50 PM
@desc:

'''
import math
import os


from  collections import  Counter
path = r'/media/isiia/TOSHIBA EXT/all_courtlistener_final_nouns_text'

conceptfreq = {}
txts = os.listdir(path)

files = ['42','20','45','41','36','30','16','44','34','1','26','5','9','21','73','69','24','7','11','10','8','31','40','18','38','75','13','71','56','6','67','19','52','28','22','14','65','12','33','68','39','15','25','78','43','4','51','62','29','66']

numlist = []

# get the all the potential domain concepts from corpus
def getTerms (): #file
    rtlist = []
    tplist = []
    for txt in txts:
            f_r = open(path + '/' + txt, 'r')
            words = f_r.read()
            if  len(words)> 0:
                wordlist = words.strip().split(" ")
                for word in wordlist:
                       tplist.append(word)
            f_r.close()

    result = Counter(tplist)
    d = sorted(result.items(), key=lambda x: x[1], reverse=True)

    for i in d:
        if i[1] > 50:
            rtlist.append(i[0])
    return rtlist



# get the frequency of  a term in the domain
def getDomainFrequency (strfile):
    tplist = []
    for txt in txts:
        if txt.split('_')[0] == strfile:
            f_r = open(path+'/'+txt,'r')
            words = f_r.read()
            if len(words) > 0:
                wordlist = words.strip().split(' ')
                for word in wordlist:                 #only save the frequency of terms in the sepecifical domain
                    if word in numlist:
                        tplist.append(word)
            f_r.close()
    return  Counter(tplist)

# get the max or min valve of terms
def getMinandMaxFre(filename):
    termsmaxin = {}
    termsmaxout = {}
    for nu in numlist:
        if conceptfreq[filename][nu] > 0:  # if term exists in the file
            nummanxin = nummaxout = 0
            for file in files:
                if file == filename:                   # get the frequency of terms in the corresponding domain corpus
                    nummanxin = conceptfreq[file][nu]
                else:
                    termnodomainfre = conceptfreq[file][nu]
                    if termnodomainfre > nummaxout:
                        nummaxout = termnodomainfre
            termsmaxin[nu] = nummanxin
            termsmaxout[nu] = nummaxout
    return termsmaxin, termsmaxout



# compute the dp of the terms
def getTermsDp(filename):
    dpdic = {}
    termsmaxin, termsmaxout = getMinandMaxFre(filename)
    for nu in numlist:
        if conceptfreq[filename][nu] > 0:  # if term exist in the file
            if termsmaxout[nu] == 0:
                dpdic[nu] = termsmaxin[nu]  # if the
            else:
                dpdic[nu] = termsmaxin[nu] * 1.0 / termsmaxout[nu]
    return dpdic


# compute the dc of the terms
def getTermsdc(filename):
    txts = os.listdir(path)
    dcdic = {}


    for nu in numlist:
        if conceptfreq[filename][nu] > 0:  # if term exists in the file
            sum = 0.0
            for txt in txts:
                if txt.split('_')[0] == filename:

                    f_tx = open(path+"/"+txt,'r')
                    words = f_tx.read().split(' ')
                    termfre = Counter(words)[nu]
                    termMaxfre = max(Counter(words).values())
                    f_tx.close()

                    if termfre != 0:
                        sum = sum + -1 * math.log(termfre / termMaxfre) * termfre / termMaxfre
            dcdic[nu] = sum
    return dcdic


if __name__ == '__main__':

    # get the potential domain concepts from the corpus, the results as the following
    f_w = open('../../tempdata/fifty_newpotentialconcepts_weights.txt', 'w')

    # get the potential terms in the legal domain
    numlist = getTerms()
    print(numlist)
    # numlist = ['reason', 'time', 'gener', 'determin']

    # compute the frequency of terms in or out in correponding domain

    for file in files:
        conceptfreq[file] = getDomainFrequency(file)

    # compute the dp and dc of each domain
    for file in files:
        # get the domain concepts candidates from the specific domain
        # get the frequency of nouns in this domain
        sum = {}
        dpdic = getTermsDp(file)
        dcdic = getTermsdc(file)

        for key, values in dpdic.items():
            sum[key] =  format(values) +"_" + format(dcdic[key])

        print(file)
        print(sum)
        f_w.write(file + "_"+ str(sum)+'\n')
    f_w.close()

