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
import ast
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import csv

import matplotlib.pyplot as plt

from sklearn import metrics
import numpy as np

import pandas
df = pandas.read_csv(r"tempdata/domainfeatures_1.csv",header=None)
acc_x = []
acc_y = []

# i= 1
# while i<=9811:
X = df.iloc[:,0:9811].values
y = df.iloc[:,9811].values
print(y)
# print(X)
# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# y = y[indices]
# print(y)
# print(len(y))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

from  collections import  Counter
print(Counter(y_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(len(y_test))

# # Decrease the dimensions using pca
from sklearn.decomposition import PCA
pca = PCA(n_components=9811)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


clf = RandomForestClassifier(n_estimators=72, random_state=0,  max_depth=16, min_samples_leaf=5)
clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print('the cross score:',scores*100)


y_pred = clf.predict(X_test)
y_t_p = clf.predict(X_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,recall_score,\
    f1_score,roc_auc_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



print('the test accuracy score:',accuracy_score(y_test, y_pred)*100)
print('the test  precision score:',precision_score(y_test, y_pred, average = 'macro')*100)
print('the test recall score:', recall_score(y_test, y_pred, average = 'macro')*100)
print('the test f1 score:', f1_score(y_test, y_pred, average = 'macro')*100)







# print('the test roc_auc_score :', roc_auc_score(y_test, y_pred, average = 'macro')*100)
#
# print('the train accuracy score:',accuracy_score(y_train, y_t_p)*100)
# print('the train  precision score:',precision_score(y_train, y_t_p, average = 'macro')*100)
# print('the train recall score:', recall_score(y_train, y_t_p, average = 'macro')*100)
# print('the train f1 score:', f1_score(y_train, y_t_p, average = 'macro')*100)



#     acc_x.append(i)
#     acc_y.append(accuracy_score(y_test, y_pred))
#     i=i+50
#
# a_x = np.array(acc_x)
# a_y = np.array(acc_y)
#
# plt.plot(a_x, a_y)
# plt.show()

# import seaborn as sns; sns








