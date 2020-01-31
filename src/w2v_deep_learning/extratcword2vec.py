# this program aims to extract word2vec from the perspective of skip or cbow

import warnings
import multiprocessing

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 忽略警告
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

if __name__ == '__main__':

    # inp为输入语料, outp1为输出模型, outp2为vector格式的模型
    inp = '/home/isiia/PycharmProjects/legal_classification/w2v_deeplearning/data/allrawdata.txt'
    out_model = '/home/isiia/PycharmProjects/legal_classification/w2v_deeplearning/data/allrawdata.model'
    out_vector = '/home/isiia/PycharmProjects/legal_classification/w2v_deeplearning/data/allrawdata_skip_300.txt'

    # sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    # size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    model = Word2Vec(LineSentence(inp), size=300, window=8, min_count=5,
                     workers=multiprocessing.cpu_count(), sg=0)

    # 保存模型
    model.save(out_model)
    # 保存词向量
    model.wv.save_word2vec_format(out_vector, binary=False)
