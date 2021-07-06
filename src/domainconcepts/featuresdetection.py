#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:21:33 2019

@author: iialab
"""

# !/usr/bin/env python
# encoding: utf-8
'''
@author: haihua
@contact: haihua.chen@unt.edu
@file:randomforests.py
@time:  7:21 PM
@desc:

# '''
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


import pandas
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



numlist = getTerms ()

df = pandas.read_csv(r"tempdata/domainfeatures_1.csv",header=None)
acc_x = []
acc_y = []

i= 9811
while i<=9811:
    X = df.iloc[:,0:i].values
    y = df.iloc[:,9811].values
    print(y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    print(y)
    print(len(y))


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print(len(y_test))


    clf = RandomForestClassifier(n_estimators=72, random_state=0,  max_depth=16, min_samples_leaf=5)
    clf.fit(X_train, y_train)

 #  detect the importance of each features
    import csv
    fcsv = open('tempdata/onehot.csv', 'w', newline='')
    spamwriter = csv.writer(fcsv)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        tpft = []
        tpft.append(numlist[indices[f]])
        # tpft.append(importances[indices[f]])
        spamwriter.writerow(tpft)
    fcsv.close()



    y_pred = clf.predict(X_test)



    from sklearn.metrics import  f1_score
    print(f1_score(y_test, y_pred, average='macro'))

    acc_x.append(i)
    acc_y.append(f1_score(y_test, y_pred, average='macro'))
    i=i+500

a_x = np.array(acc_x)
a_y = np.array(acc_y)

plt.plot(a_x, a_y)
plt.show()













