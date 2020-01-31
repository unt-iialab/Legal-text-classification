#!/usr/bin/env python
# encoding: utf-8
'''
@author: haihua
@contact: haihua.chen@unt.edu
@file:glove_sentences_lstm.py
@time:  3:09 PM
@desc: the purpose of this program is to compute classification performance based on GloVe sentence vector
@desc: This program focuses on lstm model

'''
import  csv
import numpy as np
import os

# avoid breaking off when outputting as numpy format
np.set_printoptions(linewidth= 1000000)
embeddings_index = {}
data = []
labels = []


EMBEDDING_DIM = 300
MAX_SEQENCE_LENGTH = 50
VALIDATION_SPLIT = 0.8
filetype = ['42','20','45','41','36','30','16','44','34','1','26','5','9','21','73','69','24','7','11','10','8','31','40','18','38','75','13','71','56','6','67','19','52','28','22','14','65','12','33','68','39','15','25','78','43','4','51','62','29','66']

"-------------------Get the word GloVe vectors---------------------------------"
savePath = '/home/isiia/Desktop/GloVe-1.2/newvectors.txt'
gloves_index = {}
f_glove = open(savePath, 'r')
for line in f_glove:
    stname = line.replace("\n", "").split(' ')[0]
    stvector = line.replace("\n", "").split(' ')[1:]
    gloves_index[stname] = stvector
f_glove.close()

"-------------------save the sentences GloVe vectors---------------------------------"
strPath = r"/media/isiia/TOSHIBA EXT/all_courtlistener_raw_text"
f_sg = open(r"/home/isiia/PycharmProjects/legal_classification/gloVe_deeplearning/data/glove_sentence.csv",mode='w',
            newline='')
scv_writer_vector = csv.writer(f_sg,  delimiter=' ')

f_lb = open(r"/home/isiia/PycharmProjects/legal_classification/gloVe_deeplearning/data/label.csv",mode='w',
            newline='')
scv_writer_lb = csv.writer(f_lb,  delimiter=' ')



for file in filetype:

    # the specific path of each category
    txtpath = strPath + '/' + file
    txts = os.listdir(txtpath)


    for txt in  txts:
        f_r = open(txtpath+ '/'+txt, 'r')
        lines = f_r.readlines()

        lb_list = []
        lb_list.append(filetype.index(file.split("_")[0]))

        # the paragraph mark
        k = 1
        for line in lines:
            stname = file+"_"+txt + "_" + format(k)
            tplist = []
            tplist.append(stname)
            lb_list.append(stname)
            sum = np.zeros((300,), dtype= np.float)
            num = 0
            for item in line.split(' '):
                try:
                    sum = np.add(sum,np.float_(gloves_index[item]))
                    num = num + 1
                except KeyError:
                    sum = sum +  np.zeros((300,), dtype= np.float)
                except ValueError:
                     sum = sum + np.zeros((300,), dtype=np.float)
            if num > 0:
                sum = np.divide(sum, np.ones(300)*num)
            k = k+1
            scv_writer_vector.writerow(tplist+sum.tolist())
        scv_writer_lb.writerow(lb_list)

f_sg.close()
f_lb.close()