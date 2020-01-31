path = r"/media/isiia/TOSHIBA EXT/all_courtlistener_final_nouns_text"
svepath = r"data/termpercentage_2000_frequency.txt"
f_save = open(svepath,'w')
files = ['42','20','45','41','36','30','16','44','34','1','26','5','9','21','73','69','24','7','11','10','8','31','40','18','38','75','13','71','56','6','67','19','52','28','22','14','65','12','33','68','39','15','25','78','43','4','51','62','29','66']
import  os
txts = os.listdir(path)

import pandas
from  collections import  Counter
df = pandas.read_csv("data/onehot.csv",header=None)
terms = df.iloc[:2000,0:1].values
print(terms)

for term in terms:
    tstr = ""+ term[0]
    for file in files:
        c_n = 0
        f_n = 0
        sum = 0
        for txt in txts:
            if txt.split('_')[0] ==  file:
                f_n = f_n + 1
                f_tp = open(path+'/'+txt,'r')
                words = f_tp.read().split(' ')
                f_tp.close()
                if term[0] in words:
                    c_n = c_n + 1
                    sum = sum + Counter(words)[term[0]]
        fla  = c_n * 1.0 / f_n
        tstr = tstr + " " + file  + "_" + str(fla) + "_" + str(sum) + "    "
    f_save.write(tstr+'\n')
f_save.close()


