import os
f1= os.listdir('/home/isiia/Documents/legal_classification_final_all/embeddings/bert_sentence'
            '/bert_embedding_sentence_1_78')

f2 = os.listdir('/media/isiia/TOSHIBA EXT/all_courtlistener_final_clean_text')
from  collections import  Counter
n = 0
tlist = []
for item in f2:
    f2_1 = item.split('.')[0]
    k= 1
    for it in f1:
        f1_1 = it.split('.')[0]
        if f2_1 ==f1_1:
            k =0
            break
    if k ==1:
        tlist.append(item.split('_')[0])
print(Counter(tlist))