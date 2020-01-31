import os
import nltk

nns = ['NN', 'NNS', 'NNP', 'NNPS']

path = r"/media/isiia/TOSHIBA EXT/all_courtlistener_final_clean_text"
savePath = r"/media/isiia/TOSHIBA EXT/all_courtlistener_final_nouns_text"


files = os.listdir(path)

for file in files:
    # f_w = open(savePath+ "/"+file,'w')
    f_r = open(path + "/"+ file, 'r')
    str = ""
    words = f_r.read()
    if len(words) > 0:
        lps = nltk.pos_tag(words.split(" "))

        for lp in lps:
            if lp[1] in nns:
                if len(lp[0]) > 1:
                    print(lp)

                    str = str + " " + lp[0]
        # f_w.write(str)
    f_r.close()
    # f_w.close()