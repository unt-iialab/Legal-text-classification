# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:53:32 2019

@author: iialab
"""

import os
import ast

svPath = r"/home/isiia/Desktop/bert5-9"
strPath= r"/media/isiia/TOSHIBA EXT/original-embedding-1-20"



floads = os.listdir(strPath)
savefiles = ['6','7','8','9']

for fload in sorted(floads):
    if fload in savefiles:
        print (fload)
        txts = strPath+"/"+ fload
        files = os.listdir(txts)
        for file in sorted(files):
                f = open(txts+ "/"+ file,"r")
                content = f.readlines()
                f_w = open(svPath+"/"+fload+"_"+file+".txt","w")
                for content_item in content:
                    content0 = ast.literal_eval(content_item)
                    value_feature = content0['features']
                    item = value_feature[0]
                    layers_value = item['layers']
                    item_value = layers_value[0]
                    vector = item_value['values']
                    f_w.writelines(str (vector)+ '\n')
                    print(vector)
f.close()
f_w.close()
