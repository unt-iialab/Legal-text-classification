# LeiWu 12.22
# the purpose of program is to calculate the avg_sentence and avg_words in each category

import  os
import re
path = r"/media/isiia/TOSHIBA EXT/all_courtlistener_raw_text"

selfiles = ['42','20','45','41','36','30','16','44','34','1','26','5','9','21','73','69','24','7','11','10','8','31','40','18','38','75','13','71','56','6','67','19','52','28','22','14','65','12','33','68','39','15','25','78','43','4','51','62','29','66']
alllist = []

for firfile in selfiles:
    secPath = os.listdir(path + "/" + firfile)
    sumWords = 0
    sumSentence = 0
    templist = []

    for thirfile in secPath:
        corpus = open(path + "/" + firfile+"/"+thirfile,"r")
        allwords = corpus.read()
        words = allwords.replace("\n"," ").split(" ")

        # get the number of words
        sumWords = sumWords + len(words);

        # get the number of sentence
        numSentence = 0
        # seperators = re.split('\.|\?|\!',allwords.replace("\n"," "))
        seperators = allwords.split('\n')

        if len(seperators)>1:
            numSentence = len(seperators)-1
        else:
            numSentence = len(seperators)

        sumSentence = sumSentence + numSentence
    # print(sumWords,sumSentence,len(secPath))

    templist.append(firfile)
    templist.append(sumWords/len(secPath))
    templist.append(sumSentence/len(secPath))

    alllist.append(templist)
print(alllist)




