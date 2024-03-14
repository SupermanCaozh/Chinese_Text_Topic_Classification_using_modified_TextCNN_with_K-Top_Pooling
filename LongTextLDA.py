'''
coding:utf-8
@Software:PyCharm
@Time:2023/5/12 21:58
@Author:Super Cao
'''

# 将新闻标题短文本融合为长文本

import pandas as pd
import numpy as np

# 读取当前短文本训练集
train_docs = pd.read_csv(r'toutiao_train_set.txt', sep=',', header=None)
train_docs.columns = ['text', 'category']
train_docs_lst = train_docs['text'].tolist()
# 融合每一类的文本为长文本
df_longtext_total = pd.DataFrame(columns=['text', 'category'])
keys = list(set(train_docs['category'].tolist()))
count = 0
for key in keys:
    df_temp = train_docs[train_docs['category'] == key]
    df_temp.index = list(range(df_temp.shape[0]))
    if df_temp.shape[0] % 2 == 0:
        for i in range(0, df_temp.shape[0], 2):
            text = df_temp.loc[i, 'text']
            text2 = df_temp.loc[i + 1, 'text']
            text_new = text + text2
            df_longtext_total.loc[count] = [text_new, key]
            count += 1
    if df_temp.shape[0] % 2 != 0:
        for i in range(0, df_temp.shape[0] - 1, 2):
            text = df_temp.loc[i, 'text']
            text2 = df_temp.loc[i + 1, 'text']
            text_new = text + text2
            df_longtext_total.loc[count] = [text_new, key]
            count += 1
        df_longtext_total.loc[count] = [df_temp.loc[i + 1, 'text'], key]
        count += 1

# %% 在长文本上训练LDA模型

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import numpy as np

import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import shuffle

# 读取停用词
with open(r'hit_stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = f.readlines()
stop_words_correct = []
for word in stop_words:
    stop_words_correct.append(word.strip('\n'))
# 针对新闻标题数据特点,新增新闻标题停用词
news_stop_words = ['什么', '为什么', '2018', '哪些', '一个', '怎样', '怎么样', '为何', '哪个', '到底', '还是', '如何',
                   '可以', '真的', '网友', '怎么', '中国', '美国']

n_gram2 = 2
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, n_gram2), stop_words=stop_words + news_stop_words,
                                   norm='l1',
                                   smooth_idf=True, sublinear_tf=False)

# 读取训练数据集
train_docs = pd.read_csv(r'toutiao_train_set_long.txt', sep=',', header=None)
train_docs.columns = ['text', 'category']
train_docs = shuffle(train_docs)
train_docs_lst = train_docs['text'].tolist()
train_words_lst = [' '.join(jieba.lcut(x)) for x in train_docs_lst]
# 训练tf-idf模型
tfidf_vectorizer.fit(train_words_lst)
joblib.dump(tfidf_vectorizer, 'tf_idf_vectorizer_long.pkl')

# 读取测试数据集
test_set = pd.read_csv(r'toutiao_test_set.txt', sep=',', header=None)
test_set.columns = ['text', 'category']
test_docs_lst = test_set['text'].tolist()
test_words_lst = [' '.join(jieba.lcut(x)) for x in test_docs_lst]
test_counter = tfidf_vectorizer.transform(test_words_lst)

n_topics = 15  # 即对应15个类别
while True:
    seed = np.random.randint(0, 2023)
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=100,
                                    learning_method='batch',
                                    learning_offset=50,
                                    doc_topic_prior=0.1,
                                    topic_word_prior=0.01, random_state=seed)
    lda.fit(tfidf_vectorizer.transform(train_words_lst))
    tfidf_counter = tfidf_vectorizer.transform(train_words_lst)
    plex = lda.perplexity(tfidf_counter[:10000])
    if plex < 5000000:
        break

# 将模型保存至本地
joblib.dump(lda, 'tf_idf_lda_long.model')
# 加载本地模型
# lda = joblib.load('tf_idf_lda.model')

# %% 可视化LDA模型
import pyLDAvis
import pyLDAvis.lda_model
import joblib

lda_model = joblib.load('tf_idf_lda_long.model')
tfidf_vectorizer = joblib.load('tf_idf_vectorizer_long.pkl')
tfidf_counter = tfidf_vectorizer.transform(train_words_lst)

# pyLDAvis.enable_notebook()
n_topics = 15
pic = pyLDAvis.lda_model.prepare(lda_model, tfidf_counter, tfidf_vectorizer)
# pyLDAvis.show(pic)
pyLDAvis.save_html(pic, 'lda_pass' + str(n_topics) + '.html')  # 将可视化结果打包为html文件
pyLDAvis.show(pic, local=False)
