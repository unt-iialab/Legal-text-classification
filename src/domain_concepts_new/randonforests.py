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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import plotly.express as px
from sklearn.datasets import load_boston


import pandas
eighty_df = pandas.read_csv(r"../../ccnu_data/1202_82_eighty.csv", header=None)
print(eighty_df.shape)
X_train = eighty_df.iloc[:,0:9811].values
y_train = eighty_df.iloc[:,9811].values


twenty_df = pandas.read_csv(r"../../ccnu_data/1202_82_twenty.csv", header=None)
X_test = twenty_df.iloc[:,0:9811].values
y_test = twenty_df.iloc[:,9811].values
print(twenty_df.shape)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# Decrease the dimensions using pca
from sklearn.decomposition import PCA
pca = PCA(n_components=400)
# print(pca)
pca.fit(X_train)
# print(pca.fit(X_train))
X_train = pca.transform(X_train)
# print(X_train)
X_test = pca.transform(X_test)


clf = RandomForestClassifier(n_estimators=72, random_state=0,  max_depth=10, min_samples_leaf=5)
clf.fit(X_train, y_train)
cv = KFold(n_splits=5,random_state=0, shuffle=True)
score = cross_val_score(clf,X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1)

print('Cross-validation-accuracy',score)

y_pred = clf.predict(X_test)
# y_t_p = clf.predict(X_train)

print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,recall_score,\
    f1_score,roc_auc_score
# print(confusion_matrix(y_test, y_pred))
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
