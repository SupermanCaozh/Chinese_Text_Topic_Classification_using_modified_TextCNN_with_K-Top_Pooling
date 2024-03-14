import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import pandas as pd

torch.cuda.set_device(0)

w2v = pd.read_excel(r'w2v.xlsx', header=None)
w2v = np.array(w2v)


def wv_embed(x, embedding_size):
    x2v = np.ones((len(x), x.shape[1], embedding_size))
    for i in range(len(x)):
        x2v[i] = np.array([w2v[idx] for idx in x[i]])
    return torch.tensor(x2v).to(torch.float32)


class textCNN(nn.Module):
    def __init__(self, param):
        super(textCNN, self).__init__()
        ci = 1  # input chanel size
        self.kernel_num = param['kernel_num']  # output chanel size
        self.kernel_size = param['kernel_size']
        self.vocab_size = param['vocab_size']
        self.embed_dim = param['embed_dim']
        self.dropout = param['dropout']
        self.class_num = param['class_num']
        self.param = param
        # self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, self.kernel_num, (self.kernel_size[0], self.embed_dim))
        self.conv12 = nn.Conv2d(ci, self.kernel_num, (self.kernel_size[1], self.embed_dim))
        self.conv13 = nn.Conv2d(ci, self.kernel_num, (self.kernel_size[2], self.embed_dim))
        self.conv14 = nn.Conv2d(ci, self.kernel_num, (self.kernel_size[3], self.embed_dim))
        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(len(self.kernel_size) * self.kernel_num, self.class_num)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x = torch.sigmoid(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        # TODO init embed matrix with pre-trained
        # x = self.embed(x)
        x = wv_embed(x, self.embed_dim)
        # x: (batch, sentence_length, embed_dim)
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x4 = self.conv_and_pool(x, self.conv14)
        x = torch.cat((x1, x2, x3, x4), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        # logit = F.log_softmax(self.fc1(x), dim=1)
        logit = F.softmax(self.fc1(x), dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
