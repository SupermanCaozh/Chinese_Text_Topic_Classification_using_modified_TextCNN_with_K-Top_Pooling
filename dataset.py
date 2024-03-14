'''
coding:utf-8
@Software:PyCharm
@Time:2023/5/13 3:15
@Author:Super Cao
'''
import jieba
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from tqdm import tqdm


# 读取训练用数据,并将文本转化为向量

# 构造高频词表和对应的词向量
def word2indx(words):
    wv_model = Word2Vec.load("wv_model.model")
    embedding_size = 300
    vocab = []
    wordEmbedding = []
    vocab.append("PAD")
    vocab.append("UNK")
    wordEmbedding.append(np.zeros(embedding_size))
    wordEmbedding.append(np.random.randn(embedding_size))
    for word in words:
        try:
            vector = wv_model.wv[word]
            vocab.append(word)
            wordEmbedding.append(vector)
        except:
            print(word + "不存在于词向量中")
    return vocab, np.array(wordEmbedding)


def text2ids(vocab, train_texts):
    train_ids = [[] for i in range(len(train_texts))]
    word2idx = dict(zip(vocab, list(range(len(vocab)))))
    for text in tqdm(train_texts):
        # text_word_lst = jieba.lcut(text.replace(" ", ""))
        text_lst = text.split(' ')
        for word in text_lst:
            try:
                train_ids[train_texts.index(text)].append(word2idx[word])
            except:
                # print(word + "不存在于词向量中")
                continue
    return train_ids


def get_train_data_id(vocab, train_ids, max_seq_len=20):
    reviews = []
    word2idx = dict(zip(vocab, list(range(len(vocab)))))
    for review in train_ids:
        if len(review) >= max_seq_len:
            reviews.append(review[:max_seq_len])
        else:
            reviews.append(review + [word2idx["PAD"]] * (max_seq_len - len(review)))
    return reviews


class NewsDataset(Dataset):
    def __init__(self, name, tokenizer, seq_len=16):
        super().__init__()
        self.name = name
        self.tokenizer = tokenizer
        df = pd.read_csv('{}.txt'.format(name), sep=',',
                         names=['text', 'label'])
        self.txt = list(df.text)
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
        self.label = list(df.label)
        self.seq_len = seq_len

    def __getitem__(self, idx):
        return torch.tensor(
            self.tokenizer(self.txt[idx], max_len=self.seq_len)
        ), np.int64(self.label[idx])

    def __len__(self):
        return len(self.label)
