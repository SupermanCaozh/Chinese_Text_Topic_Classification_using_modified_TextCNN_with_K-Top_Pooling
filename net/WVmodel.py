'''
coding:utf-8
@Software:PyCharm
@Time:2023/5/13 13:04
@Author:Super Cao
'''
import gensim
import pandas as pd
import jieba
from gensim.models import Word2Vec
import joblib

# 读取语料库
train_docs = pd.read_csv(r"D:\研究生课件\大数据分析\FinalReoport\toutiao_train.txt", sep=',')
train_docs.columns = ['text', 'label']
train_docs_lst = train_docs['text'].tolist()

sentences = [jieba.lcut(i.replace(" ", "")) for i in train_docs_lst]

model = Word2Vec(epochs=10, min_count=1, vector_size=300, window=5)
model.build_vocab(sentences)
model.train(sentences, total_examples=len(sentences), epochs=5)

model.save('wv_model.model')