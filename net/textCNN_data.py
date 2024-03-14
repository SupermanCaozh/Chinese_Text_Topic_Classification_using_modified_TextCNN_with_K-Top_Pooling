import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np
from dataset import *
from collections import Counter

# trainDataFile = 'traindata_vec.txt'
trainDataFile = r"D:\研究生课件\大数据分析\FinalReoport\toutiao_train.txt"
valDataFile = 'valdata_vec.txt'

# class_map = {'100': 0, '101': 1, '102': 2, '103': 3, '104': 4, '106': 5, '107': 6, '108': 7, '109': 8, '110': 9,
#              '112': 10, '113': 11, '114': 12, '115': 13, '116': 14}
class_map = {'102': 0, '103': 1, '107': 2, '109': 3, '116': 4}


def get_valdata(file=valDataFile):
    valData = open(valDataFile, 'r').read().split('\n')
    valData = list(filter(None, valData))
    random.shuffle(valData)

    return valData


def min_max_norm(text_vec):
    min_value = np.min(text_vec)
    max_value = np.max(text_vec)
    norm_text_vec = (text_vec - min_value) / (max_value - min_value)
    return text_vec


class textCNN_data(Dataset):
    def __init__(self):
        # trainData = open(trainDataFile, 'r').read().split('\n')
        # trainData = list(filter(None, trainData))
        # random.shuffle(trainData)
        # self.trainData = trainData

        # 用Word2Vec词向量当特征
        trainData = pd.read_csv(trainDataFile, sep=',', header=None)
        trainData.columns = ['text', 'label']
        all_texts = trainData['text'].tolist()[:50000]
        self.all_texts = all_texts
        self.all_labels = trainData['label'].tolist()

        all_texts_cut = [' '.join(jieba.lcut(x)) for x in all_texts]
        all_words = ' '.join(all_texts_cut).split(' ')
        # 统计词频
        wordCount = Counter(all_words)
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sortWordCount if item[1] >= 10]

        vocab, wvembedding = word2indx(words)
        self.vocab = vocab
        self.wvembedding = wvembedding
        # 存储词表
        with open('vocab.txt', 'w') as f:
            f.write(str(vocab))
        # 存储嵌入的词向量
        wv_df = pd.DataFrame(wvembedding)
        wv_df.to_excel('w2v.xlsx', index=False, header=False)

        self.train_ids = text2ids(self.vocab, all_texts_cut)
        self.reviews = get_train_data_id(self.vocab, self.train_ids)

    def __len__(self):
        # return len(self.trainData)
        return len(self.all_texts)

    def __getitem__(self, idx):
        # data = self.trainData[idx]
        # data = list(filter(None, data.split(',')))
        # data = [int(x) for x in data]
        # cla = class_map[str(data[0])]
        # # sentence = np.array(data[1:])
        # sentence = min_max_norm(np.array(data[1:11]))
        data_id = self.reviews[idx]
        # embedding
        # sentence = []
        # for id in data_id:
        #     sentence.append(self.wvembedding[id])
        # sentence = np.array(sentence)
        cla = class_map[str(self.all_labels[idx])]

        return cla, np.array(data_id)


def textCNN_dataLoader(param):
    dataset = textCNN_data()
    batch_size = param['batch_size']
    shuffle = param['shuffle']
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    dataset = textCNN_data()
    for i in range(100):
        cla, sen = dataset.__getitem__(i)

        print(cla)
        print(sen)
