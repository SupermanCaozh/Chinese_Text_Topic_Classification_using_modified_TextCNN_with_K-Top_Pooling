'''
coding:utf-8
@Software:PyCharm
@Time:2023/5/14 15:05
@Author:Super Cao
'''
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from dataset import *
import pandas as pd
from model import textCNN
import jieba
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class_map = {'102': 0, '103': 1, '107': 2, '109': 3, '116': 4}

textCNN_param = {
    # 'vocab_size': len(word2ind),
    'vocab_size': 300,
    'embed_dim': 300,
    'class_num': 5,
    "kernel_num": 16,
    "kernel_size": [3, 4, 5, 6],
    "dropout": 0.5,
}
net = textCNN(textCNN_param)
weightfp = r'model_normal\23051417_model_iter_49_390_loss_1.51.pkl'
print('load weight')
net.load_state_dict(torch.load(weightfp))

net.eval()

# 获得预测概率
test_docs = pd.read_csv(r"D:\研究生课件\大数据分析\FinalReoport\toutiao_test.txt", sep=',', header=None)
test_docs.columns = ['text', 'label']
test_texts = test_docs['text'].tolist()[:10000]
test_labels = test_docs['label'].tolist()[:10000]
bi_labels = [class_map[str(x)] for x in test_labels]

with open('vocab.txt', 'r') as f:
    str_vocab = f.readlines()[0]
vocab = eval(str_vocab)
wvembedding = np.array(pd.read_excel(r'w2v.xlsx', header=None))


def w2v(sentence, vocab, wvembedding):
    words = jieba.lcut(sentence)
    vec_idx = []
    for word in words:
        if word in vocab:
            vec_idx.append(vocab.index(word))
        else:
            continue
    if len(vec_idx) > 20:
        vec_idx = vec_idx[:21]
    elif len(vec_idx) < 20:
        vec_idx = vec_idx + [0] * (20 - len(vec_idx))
    # vec = []
    # for idx in vec_idx:
    #     vec_temp = wvembedding[idx]
    #     vec.append(vec_temp)
    # vec = np.array(vec)
    return np.array([vec_idx])


def single_category_roc_info(cat):
    true_label = []
    pred_prob = []
    for text in test_texts:
        idx = test_texts.index(text)
        label = test_labels[idx]
        vec = w2v(text, vocab, wvembedding)
        pred = net(vec).detach().numpy()
        if label == eval(cat):  # cat为正类
            true_label.append(1)
            label_idx = class_map[str(label)]
            pred_prob_label = pred[0, label_idx]
            pred_prob.append(pred_prob_label)
        else:
            true_label.append(0)
            label_idx = class_map[cat]
            pred_prob_label = pred[0, label_idx]  # 可修改
            pred_prob.append(pred_prob_label)
    return true_label, pred_prob


true_labels = []
pred_probs = []
for cls, cls_label in tqdm(class_map.items()):
    # 分别计算每个类别的ROC并绘制曲线
    true_label, pred_prob = single_category_roc_info(cls)
    true_labels.append(true_label)
    pred_probs.append(pred_prob)

colors = ['red', 'blue', 'green', 'purple', 'orange']
style = ['-.', '--', 'solid', ':', '-']
cat_labels = ['娱乐', '体育', '汽车', '科技', '电竞']
fig = plt.figure()
for i in range(5):
    fpr, tpr, thresholds = roc_curve(true_labels[i], pred_probs[i])
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, c=colors[i], ls=style[i], label=cat_labels[i] + f"(area={area:.2f})")
plt.grid('--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.show()


def single_category_pred_label(cat):
    true_label = []
    pred_labels = []
    for text in test_texts:
        idx = test_texts.index(text)
        label = test_labels[idx]
        vec = w2v(text, vocab, wvembedding)
        pred = net(vec).detach().numpy()
        if label == eval(cat):  # cat为正类
            true_label.append(1)
            label_idx = class_map[str(label)]
            pred_label = np.argmax(pred[0])
            if pred_label == label_idx:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        else:
            true_label.append(0)
            label_idx = class_map[cat]
            pred_label = np.argmax(pred[0])
            if pred_label == label_idx:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
    return true_label, pred_labels


def get_pred_label():
    true_label = []
    pred_labels = []
    for text in test_texts:
        idx = test_texts.index(text)
        label = test_labels[idx]
        label_idx = class_map[str(label)]
        true_label.append(label_idx)
        vec = w2v(text, vocab, wvembedding)
        pred = net(vec).detach().numpy()
        pred_label = np.argmax(pred[0])
        pred_labels.append(pred_label)
    return true_label, pred_labels


# 计算precise,recall,f1-score等评价指标
true_labels = []
pred_labelss = []
for cls, cls_label in tqdm(class_map.items()):
    true_label, pred_labels = single_category_pred_label(cls)
    true_labels.append(true_label)
    pred_labelss.append(pred_labels)
    pre= precision_score(true_label, pred_labels, average='binary')
    print(f"{cls}类别: Precision: {pre}")

true_label, pred_labels = get_pred_label()
print(precision_score(true_label, pred_labels, average='micro'))
