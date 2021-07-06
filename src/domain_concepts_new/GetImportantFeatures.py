
import ast
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import csv

import matplotlib.pyplot as plt

from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import plotly.express as px
from sklearn.datasets import load_boston
from collections import Counter

filetype = ['42','20','45','41','36','30','16','44','34','1','26','5','9','21','73','69','24','7','11','10','8','31',
         '40','18','38','75','13','71','56','6','67','19','52','28','22','14','65','12','33','68','39','15','25','78','43','4','51','62','29','66']

path = r'/media/junhua/TOSHIBA EXT/all_courtlistener_final_nouns_text/'
txts = os.listdir(path)

alpha = 0.8
beta = 0.2

#  get the all the potential domain concepts from corpus
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

def getMaxDPDc():
    filesmaxvalue = {}
    fr_features = open('../../tempdata/fifty_newpotentialconcepts_weights.txt', 'r')
    lines = fr_features.readlines()
    for line in lines:
        tempdp = []
        tempdc = []
        fileName = line.split('_')[0]
        dic = ast.literal_eval('{' + line.split('{')[1])
        for cp in dic.keys():
             tempdp.append(float(dic[cp].split('_')[0]))
             tempdc.append(float(dic[cp].split('_')[1]))
        filesmaxvalue[fileName] = str(max(tempdp))+"_"+str(max(tempdc))
    return filesmaxvalue

# calculating the vector of each file, and results saved using csv format
def setSavefiles(importantConcepts,filesmaxvalue):
    fr_features = open('../../tempdata/fifty_newpotentialconcepts_weights.txt', 'r')
    lines = fr_features.readlines()
    with open('../../ccnu_data/importantConcepts_0705', 'w', newline='') as csvfile:  # revised
        spamwriter = csv.writer(csvfile)
        tname = ['leixing']
        tname.extend(importantConcepts)
        spamwriter.writerow(tname)
        for line in lines:
            tplist = []
            dic = ast.literal_eval('{'+line.split('{')[1])
            fileName = line.split('_')[0]
            tplist.append(fileName)
            for cp in importantConcepts:
                score = 0
                if cp in dic.keys():
                    print(filesmaxvalue[fileName])
                    score = alpha * float(dic[cp].split('_')[0])/float(filesmaxvalue[fileName].split('_')[0]) + beta * float(dic[cp].split('_')[1])/float(filesmaxvalue[fileName].split('_')[1])
                    tplist.append(score)
                else:
                    tplist.append(score)
            spamwriter.writerow(tplist)
    fr_features.close()
    return

import pandas
eighty_df = pandas.read_csv(r"../../ccnu_data/1202_82_eighty.csv", header=None)
X_train = eighty_df.iloc[:,0:9811].values
y_train = eighty_df.iloc[:,9811].values


twenty_df = pandas.read_csv(r"../../ccnu_data/1202_82_twenty.csv", header=None)
X_test = twenty_df.iloc[:,0:9811].values
y_test = twenty_df.iloc[:,9811].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Decrease the dimensions using pca
from sklearn.decomposition import PCA
pca = PCA(n_components=400)
pca.fit(X_train)
X_train = pca.transform(X_train)

n_pcs = pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

concepts = getTerms()
most_important_concepts = [concepts[most_important[i]] for i in range(n_pcs)]

print(most_important_concepts)

filesmaxvalue = getMaxDPDc()
setSavefiles(most_important_concepts,filesmaxvalue)







