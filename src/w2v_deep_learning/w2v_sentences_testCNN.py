#!/usr/bin/env python
# encoding: utf-8
'''
@author: haihua
@contact: haihua.chen@unt.edu
@file:glove_sentences_lstm.py
@time:  3:09 PM
@desc: the purpose of this program is to compute classification performance based on word2vec sentence vector
@desc: This program focuses on lstm model

'''
from keras.callbacks import EarlyStopping

import csv
import numpy as np
import matplotlib.pyplot as pyplot
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Flatten, BatchNormalization
from keras.layers import Dropout, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
import keras
from keras import backend as K
import keras_metrics as km
from keras import metrics
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold

# avoid breaking off when outputting as numpy format
np.set_printoptions(linewidth=1000000)
embeddings_index = {}
data = []
labels = []

EMBEDDING_DIM = 300
MAX_SEQENCE_LENGTH = 50
VALIDATION_SPLIT = 0.4

"-------------------Get the word2vec sentence vectors---------------------------------"
f_glove = open(r"data/cbow_sentence_300.csv", 'r')
csv_reader_gl = csv.reader(f_glove, delimiter=' ')
for row in csv_reader_gl:
    sentence = row[0]
    embeddings_index[row[0]] = np.asarray(row[1:], dtype='float32')
f_glove.close()

"-----------------------get the labels and content-----------------------------------------"
f_lb = open(r"data/label.csv", 'r')
csv_reader_lb = csv.reader(f_lb, delimiter=' ')

for row in csv_reader_lb:
    labels.append(row[0])
    data.append(row[1:])
f_lb.close()

"------------------Save x_train & y_train-----------------------------"
all_pf = [0, 0, 0, 0]

labels_1 = to_categorical(labels)

tokenizer = keras.preprocessing.text.Tokenizer(num_words=None)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
stname_index = tokenizer.word_index

data_1 = keras.preprocessing.sequence.pad_sequences(sequences, MAX_SEQENCE_LENGTH)

indices = np.arange(data_1.shape[0])
np.random.shuffle(indices)
data_1 = data_1[indices]
labels_1 = labels_1[indices]

# nb = int(VALIDATION_SPLIT * data_1.shape[0])
# x_train = data_1[:-nb]
# y_train = labels_1[:-nb]
#
# x_val = data_1[-nb:]
# y_val = labels_1[-nb:]

"-------------------Calculate ebedding matrix---------------------------------"
enbedding_matrix = np.random.random((len(stname_index) + 1, EMBEDDING_DIM))

for stname, i in stname_index.items():
    embedding_vector = embeddings_index.get(stname)
    if embedding_vector is not None:
        enbedding_matrix[i] = embedding_vector

"------------------------textCNN　model-------------------------------"
kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)
for i, (train_index, val_index) in enumerate(kf.split(data_1, labels_1.argmax(1))):
    x_train, x_val = data_1[train_index], data_1[val_index]
    y_train, y_val = labels_1[train_index], labels_1[val_index]

    model = Sequential()
    model.add(Embedding(trainable=False, input_dim=len(stname_index) + 1, output_dim=EMBEDDING_DIM,
                        weights=[enbedding_matrix], input_length=MAX_SEQENCE_LENGTH))
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  # (批)规范化层

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='softmax'))

    earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='max')
    SGD = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])
    # metrics=['acc', km.binary_precision(),   km.binary_recall()])

    # print(model.summary())
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=78, batch_size=168, callbacks=[
        earlyStopping])
    # loss, accuracy,  precision, recall = model.evaluate(x_val, y_val, verbose=0,batch_size=32)

    y_pred = model.predict(x_val)


    a_score = accuracy_score(y_val.argmax(axis=1), y_pred.argmax(axis=1))
    p_score = precision_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    r_score = recall_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    f_score = f1_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average='macro')


    all_pf[0] = all_pf[0] + a_score
    all_pf[1] = all_pf[1] + p_score
    all_pf[2] = all_pf[2] + r_score
    all_pf[3] = all_pf[3] + f_score


    #

    # pyplot.plot(history.history['acc'], label='train')
    # pyplot.plot(history.history['val_acc'], label='test')
    # pyplot.legend()
    # pyplot.show()

print("This is 5-fold cross-validation results:")
print("accuracy_score: %.2f%%" % (all_pf[0] / 5 * 100))
print("Precision: %.2f%%" % (all_pf[1] / 5 * 100))
print("Recall: %.2f%%" % (all_pf[2] / 5 * 100))
print("f1_score: %.2f%%" % (all_pf[3] / 5 * 100))
