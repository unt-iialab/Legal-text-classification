
# '''
import ast
import os
from collections import Counter
import csv
import random
alpha = 0.8
beta = 0.2
filetype = ['42','20','45','41','36','30','16','44','34','1','26','5','9','21','73','69','24','7','11','10','8','31',
         '40','18','38','75','13','71','56','6','67','19','52','28','22','14','65','12','33','68','39','15','25','78','43','4','51','62','29','66']

path = r'/media/isiia/TOSHIBA EXT/all_courtlistener_final_nouns_text'
txts = os.listdir(path)


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

cplist = getTerms()

# randomly save terms in list
random.shuffle(cplist)

# calculating the vector of each file, and results saved using csv format
fr_features = open('tempdata/fifty_newpotentialconcepts_weights.txt','r')
lines = fr_features.readlines()
with open('tempdata/onehot.csv', 'w',  newline='') as csvfile:  #  revised
    spamwriter = csv.writer(csvfile)

    for line in lines:
        dic = ast.literal_eval('{'+line.split('{')[1])
        fileName = line.split('_')[0]
        for txt in txts:
            f_tx = open(path+'/'+txt,'r')
            words = f_tx.read().split(' ')
            if txt.split('_')[0] == fileName and txt.split('_')[0] in filetype:                    # to get some
                tplist = []
                for cp in cplist:
                    weight = 0
                    if cp in dic.keys():
                        # weight = Counter(words)[cp] * (
                        #             alpha * float(dic[cp].split('_')[0]) + beta * float(dic[cp].split('_')[1]))
                        if Counter(words)[cp]>1:
                            weight = 1

                    tplist.append(weight)
                tplist.append(int(fileName))
                spamwriter.writerow(tplist)
            f_tx.close()
fr_features.close()
