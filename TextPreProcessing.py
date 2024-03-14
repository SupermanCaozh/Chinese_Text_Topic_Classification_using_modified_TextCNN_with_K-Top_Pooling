'''
coding:utf-8
@Software:PyCharm
@Time:2023/5/9 16:17
@Author:Super Cao
'''

# 利用LDA模型进行文本主题分类

# %%
import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import matplotlib.pyplot as plt


# %%
def extrat_content_label(text):
    return [text.split('_!_')[3], text.split('_!_')[1]]


def is_chinese(uchar):
    return uchar >= u'\u4e00' and uchar <= u'\u9fa5'


def split_words(text):
    words_lst = jieba.lcut(text)
    words_str = "".join([word for word in words_lst if word not in stop_words])
    normal_str = "".join(s for s in words_str if is_chinese(s) or s in string.ascii_letters or s in string.digits)
    return normal_str


# 读取停用词
with open(r'hit_stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = f.readlines()
stop_words_correct = []
for word in stop_words:
    stop_words_correct.append(word.strip('\n'))
# 读取原始数据
with open(r'toutiao_cat_data.txt', 'r', encoding='utf-8') as f:
    corpus = f.readlines()
corpus = [extrat_content_label(x) for x in corpus]
df_raw = pd.DataFrame(corpus, columns=['text', 'category'])

# 对raw_data进行分词操作
df_raw['text'] = df_raw['text'].apply(split_words)

# 查看样本的均衡性,绘制各类新闻的直方图
news_num = df_raw.groupby('category').category.size().reset_index(name='news_num')
fig = plt.figure()
plt.bar(list(range(15)), news_num['news_num'].tolist())
plt.xticks(news_num['category'])
plt.show()

# 查看新闻标题的长度范围
# 统计值


# 构造train_set,validation_set,test_set
X_train, X_test, Y_train, Y_test = train_test_split(df_raw['text'].tolist(), df_raw['category'].tolist(),
                                                    test_size=1 / 5, random_state=516, shuffle=True)
# 写入csv文件
# 头条新闻数据训练集
# 头条新闻数据集共有382688条新闻标题数据
train_data = [list(x) for x in zip(X_train, Y_train)]
train_set = pd.DataFrame(train_data, columns=['text', 'category'])
train_set.to_csv(r'toutiao_train_set.txt', sep=',', header=False, index=False)

# 头条新闻数据测试集
test_data = [list(x) for x in zip(X_test, Y_test)]
test_set = pd.DataFrame(test_data, columns=['text', 'category'])
test_set.to_csv(r'toutiao_test_set.txt', sep=',', header=False, index=False)

# %%
# LDA模型进行短文本主题分类
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import pandas as pd

# 待讨论的超参
n_gram = 2
# 构造词袋模型,将文本向量化
vectorizer = CountVectorizer(max_df=0.5, min_df=0.01, ngram_range=(1, n_gram), max_features=None, binary=False)
# vectorizer = CountVectorizer(max_df=0.5, min_df=3, ngram_range=(1, n_gram), max_features=None, binary=False,
#                              stop_words=stop_words)
# 读取训练集数据
train_docs = pd.read_csv(r'toutiao_train_set.txt', sep=',', header=None)
train_docs.columns = ['text', 'category']
train_docs_lst = train_docs['text'].tolist()
train_words_lst = [' '.join(jieba.lcut(x)) for x in train_docs_lst]
# 训练词袋模型
vectorizer.fit(train_words_lst)

# %%
# 利用训练好的词袋模型将测试数据集文本向量化,并降维绘制网络图
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import jieba
import matplotlib.pyplot as plt


def tsne_plot(vectors, color):
    """ vectors是降维后的二维向量"""
    x = vectors[:, 0]
    y = vectors[:, 1]
    plt.scatter(x, y, c=color)


test_set = pd.read_csv(r'toutiao_test_set.txt', sep=',', header=None)
test_set.columns = ['text', 'category']

# 训练好的词袋模型用于测试集
test_docs_lst = test_set['text'].tolist()
test_words_lst = [' '.join(jieba.lcut(x)) for x in test_docs_lst]
test_counter = vectorizer.transform(test_words_lst)
# test_counter_arr = test_counter.toarray()

# 将文本类别整理成列表
cats = list(set(test_set['category'].tolist()))

# fig = plt.figure(dpi=200)
# # 定义不同category散点的颜色
# colors = ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000', '#95d0fc', '#029386', '#f97306', '#96f97b',
#           '#c20078', '#ffff14', '#75bbfd', '#929591', '#89fe05']
# for cat in cats:
#     cat_index = test_set[test_set['category'] == cat].index.to_list()
#     cat_high_vector = test_counter[cat_index].toarray()
#     # 先用pca降维
#     pca = PCA(n_components=30)
#     pca_vec = pca.fit_transform(cat_high_vector)
#     # 再用tsne降维
#     tsner = TSNE(n_components=2, learning_rate=100, random_state=516)
#     high2low = tsner.fit_transform(pca_vec)
#     tsne_plot(high2low, colors[cats.index(cat)])
# plt.savefig(r'tsne_clusters.png')
# plt.show()

# %% 利用LDA进行文本主题分类
from sklearn.decomposition import LatentDirichletAllocation

n_topics = 15  # 即对应15个类别
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=100,
                                learning_method='batch',
                                learning_offset=50,
                                doc_topic_prior=0.1,
                                topic_word_prior=0.01,
                                random_state=516)
lda.fit(vectorizer.transform(train_words_lst))


# 效果不好,词频反应的信息不准确

# %% 查看每个主题的关键词
def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):  # components打印模型学习结果
        print(f"Topic #{topic_idx}:")
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword


# 输出每个主题对应词语
n_top_words = 25
bow_feature_names = vectorizer.get_feature_names_out()
topic_word = print_top_words(lda, bow_feature_names, n_top_words)

# %% 训练tf-idf模型
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import jieba
import matplotlib.pyplot as plt

import joblib
from sklearn.decomposition import LatentDirichletAllocation

# 读取停用词
with open(r'hit_stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = f.readlines()
stop_words_correct = []
for word in stop_words:
    stop_words_correct.append(word.strip('\n'))
# 针对新闻标题数据特点,新增新闻标题停用词
news_stop_words = ['什么', '为什么', '2018', '哪些', '一个', '怎样', '怎么样', '为何', '哪个', '到底', '还是', '如何',
                   '可以', '真的', '网友', '怎么', '中国', '美国']

n_gram2 = 3
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, n_gram2), stop_words=stop_words + news_stop_words,
                                   norm='l1',
                                   smooth_idf=True, sublinear_tf=False)

# 读取训练数据集
train_docs = pd.read_csv(r'toutiao_train_set.txt', sep=',', header=None)
train_docs.columns = ['text', 'category']
train_docs_lst = train_docs['text'].tolist()
train_words_lst = [' '.join(jieba.lcut(x)) for x in train_docs_lst]
# 训练tf-idf模型
tfidf_vectorizer.fit(train_words_lst)
joblib.dump(tfidf_vectorizer, 'tf_idf_vectorizer_ngram3.pkl')

# 读取测试数据集
test_set = pd.read_csv(r'toutiao_test_set.txt', sep=',', header=None)
test_set.columns = ['text', 'category']
test_docs_lst = test_set['text'].tolist()
test_words_lst = [' '.join(jieba.lcut(x)) for x in test_docs_lst]
test_counter = tfidf_vectorizer.transform(test_words_lst)

n_topics = 15  # 即对应15个类别
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=10,
                                doc_topic_prior=0.2,
                                topic_word_prior=0.01, random_state=123)
lda.fit(tfidf_vectorizer.transform(train_words_lst))

# 将模型保存至本地
joblib.dump(lda, 'tf_idf_lda_ngram3.model')
# 加载本地模型
# lda = joblib.load('tf_idf_lda.model')

# %% 可视化LDA模型
import pyLDAvis
import pyLDAvis.lda_model

lda_model = joblib.load('tf_idf_lda_ngram3.model')
tfidf_vectorizer = joblib.load('tf_idf_vectorizer_ngram3.pkl')
tfidf_counter = tfidf_vectorizer.transform(train_words_lst)

# pyLDAvis.enable_notebook()
n_topics = 15
pic = pyLDAvis.lda_model.prepare(lda_model, tfidf_counter, tfidf_vectorizer)
# pyLDAvis.show(pic)
pyLDAvis.save_html(pic, 'lda_pass' + str(n_topics) + '.html')  # 将可视化结果打包为html文件
pyLDAvis.show(pic, local=False)
# %% 利用标签联合评价LDA模型
import pandas as pd
import numpy as np
import joblib
import jieba

# 加载训练好的模型
lda_model = joblib.load('tf_idf_lda.model')
tfidf_vectorizer = joblib.load('tf_idf_vectorizer.pkl')
# 读取训练集
train_docs = pd.read_csv(r'toutiao_train_set.txt', sep=',', header=None)
train_docs.columns = ['text', 'category']
train_docs_lst = train_docs['text'].tolist()
# 获取分词向量
train_words_lst = [' '.join(jieba.lcut(x)) for x in train_docs_lst]
tfidf_counter = tfidf_vectorizer.transform(train_words_lst)
topics = lda_model.transform(tfidf_counter)

# 统计每个topic下面的最大概率的类别中,各标签类别的个数,以此确定该话题属于哪个类别
keys = list(set(train_docs['category'].tolist()))
# single_cat_counter = dict(zip(keys, [0 for i in range(15)]))
lda_train_topic_cat = [dict(zip(keys, [0 for j in range(15)])) for i in range(15)]  # 每一行代表一个topic,每一列代表标签的个数
# 看每个带标签的样本,被lda模型归为哪一类
for i in range(len(train_docs_lst)):
    this_cat = train_docs.loc[i, 'category']
    topic = topics[i]
    prob_max_topic = max(topic)
    prob_max_index = topic.tolist().index(prob_max_topic)  # 概率最大的类别
    lda_train_topic_cat[prob_max_index][this_cat] += 1
for i in range(15):
    print(lda_train_topic_cat[i], '\n')


# 上述想法结果：由于给类的样本数量不完全一致,故受样本数量影响,无效

def cal_dist(dist_arr):
    edge_count = 0
    dist = []
    for i in range(len(dist_arr)):
        node_temp = dist_arr[i]
        for j in range(i + 1, len(dist_arr)):
            node = dist_arr[j]
            # dis = np.sqrt((node_temp - node) ** 2)
            dis = np.dot(node, node_temp) / (np.linalg.norm(node) * np.linalg.norm(node_temp))  # 余弦距离
            dist.append(dis)
            edge_count += 1
    return np.mean(dist)


# 计算每一个topic中的各类别的样本的距离
lda_train_topic_dist = [dict(zip(keys, [[] for i in range(15)])) for j in range(15)]
for i in range(len(train_docs_lst)):
    this_cat = train_docs.loc[i, 'category']
    topic = topics[i]
    prob_max_topic = max(topic)
    prob_max_index = topic.tolist().index(prob_max_topic)  # 概率最大的类别
    lda_train_topic_dist[prob_max_index][this_cat].append(topic)
# 记录每一类别的平均距离
lda_train_topic_mean_dist = [dict(zip(keys, [0 for j in range(15)])) for i in range(15)]
for i in range(len(lda_train_topic_dist)):
    for cat in keys:
        temp_arr = lda_train_topic_dist[i][cat]
        mean_cat_dist = cal_dist(temp_arr)
        lda_train_topic_mean_dist[i][cat] = mean_cat_dist
# %% 截断raw_data,即选择五类主题及其对应的数据
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

raw_data_fp = r"D:\研究生课件\大数据分析\FinalReoport\toutiao_cat_data.txt"
with open(raw_data_fp, 'r', encoding='utf-8') as f:
    corpus = f.readlines()
corpus = [extrat_content_label(x) for x in corpus]
df_raw = pd.DataFrame(corpus, columns=['text', 'label'])
# 选择的主题类别包含以下:
# 102
# 103
# 107
# 109
# 116
chosen_category = [102, 103, 107, 109, 116]
df_chosen = df_raw[
    (df_raw['label'] == '102') | (df_raw['label'] == '103') | (df_raw['label'] == '107') | (
            df_raw['label'] == '109') | (df_raw[
                                             'label'] == '116')]
df_chosen = shuffle(df_chosen)

# 画各类主题新闻文本数据量
# fig = plt.figure()
# p1 = plt.bar(list(range(5)), df_chosen.groupby('label').size().tolist())
# plt.bar_label(p1, label_type='edge')
# plt.xticks(list(range(5)), ['娱乐', '体育', '汽车', '科技', '电竞'])
# plt.show()

# 将选择的五类新闻文本进行分词操作并划分为测试集和训练集
df_chosen['text'] = df_chosen['text'].apply(split_words)

# 构造train_set,validation_set,test_set
X_train, X_test, Y_train, Y_test = train_test_split(df_chosen['text'].tolist(), df_chosen['label'].tolist(),
                                                    test_size=1 / 5, random_state=516, shuffle=True)
# 写入csv文件
# 头条新闻数据训练集
# 头条新闻数据集共有382688条新闻标题数据
train_data = [list(x) for x in zip(X_train, Y_train)]
train_set = pd.DataFrame(train_data, columns=['text', 'label'])
train_set.to_csv(r'toutiao_train.txt', sep=',', header=False, index=False)

# 头条新闻数据测试集
test_data = [list(x) for x in zip(X_test, Y_test)]
test_set = pd.DataFrame(test_data, columns=['text', 'label'])
test_set.to_csv(r'toutiao_test.txt', sep=',', header=False, index=False)