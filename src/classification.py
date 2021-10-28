import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
import xlrd
import re
import numpy as np
# coding = gbk

jieba.load_userdict("user_dict.txt")
jieba.load_userdict("dict/SogouLabDic.txt")
jieba.load_userdict("dict/dict_baidu_utf8.txt")
jieba.load_userdict("dict/dict_pangu.txt")
jieba.load_userdict("dict/dict_sougou_utf8.txt")
jieba.load_userdict("dict/dict_tencent_utf8.txt")
jieba.load_userdict("dict/my_dict.txt")

def stopwordslist(stopwords_filepath):
    stopwords = [line.strip() for line in open(stopwords_filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def clear_line(line:str):
    if(line != ''):
        line = line.strip()
        # 去除表情
        line = re.sub(u'[\U00010000-\U0010ffff]', '', line)
        line = re.sub(u'[\uD800-\uDBFF][\uDC00-\uDFFF]', '', line)
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）️《》↓\-【】♠➕●❤á✖ผิดไหม丨ท ีฉนกลับปรักเธอ\[\]]+", "", line)
        return line
    return None

def sent2word(line, stopwords):
    segList = jieba.cut(line, cut_all=False)
    segSentence = ''
    for word in segList:
        if word != '\t' and word not in stopwords:
            segSentence += word+" "
    return segSentence.strip()

def read_data(file_dir='微博博文分类', output='clean.txt'):
    f = open(output, 'w', encoding='utf-8')
    stopwords = stopwordslist('stopwords.txt')
    # print(stopwords)
    labels = []
    for root, dirs, files in os.walk(file_dir):
        i = 0
        for file in files:
            print(file)
            data = xlrd.open_workbook(root + '/' + file)
            table = data.sheet_by_name('Sheet1')  # 通过名称来获取指定页
            nrows = table.nrows  # 为行数，整形
            ncolumns = table.ncols  # 为列数，整形
            for row in range(nrows):
                if table.row_values(row)[1] == '微博内容':
                    continue
                line = table.row_values(row)[1] + ' ' + table.row_values(row)[2]
                line = clear_line(line)
                seg_line = sent2word(line, stopwords)
                f.writelines(seg_line+'\n')
                labels.append(i)
            i += 1
    np.save('labels.npy', labels)

def labels_to_original(labels, forclusterlist):
    assert len(labels) == len(forclusterlist)
    maxlabel = max(labels)
    numberlabel = [i for i in range(0, maxlabel + 1, 1)]
    numberlabel.append(-1)
    result = [[] for i in range(len(numberlabel))]
    for i in range(len(labels)):
        index = numberlabel.index(labels[i])
        result[index].append(forclusterlist[i])
    return result

def classify(path, class_num):
    corpus = []
    txt = open(path, 'r', encoding='utf-8')
    for str in txt:
        corpus.append(str)
    print('微博数目:{}'.format(len(corpus)))
    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    print('计算tfidf')
    vectorizer = CountVectorizer(max_features=30000)
    # 该类会统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    # 将文本转为词频矩阵并计算tf-idf
    tfidf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    tfidf_matrix = tfidf.toarray()
    print(tfidf_matrix.shape)
    np.save('tfidf_matrix.npy', tfidf_matrix)
    # 获取词袋模型中的所有词语
    #word = vectorizer.get_feature_names()
    # # 统计词频
    print('开始聚类')
    clf = KMeans(n_clusters=class_num)
    s = clf.fit(tfidf_matrix)
    # 每个样本所属的簇
    label = []
    i = 1
    while i <= len(clf.labels_):
        label.append(clf.labels_[i - 1])
        i = i + 1
    # 获取标签聚类
    for i in range(len(label) // 10):
        print(label[i * 10:(i + 1) * 10])

    y_pred = clf.labels_
    np.save("y_pred.npy", y_pred)
    # pca降维，将数据转换成二维
    print('PCA降维')
    pca = PCA(n_components=2)  # 输出两维
    newData = pca.fit_transform(tfidf_matrix)  # 载入N维

    xs, ys = newData[:, 0], newData[:, 1]
    # 设置颜色
    cluster_colors = {0: 'red', 1: 'yellow', 2: 'blue', 3: 'green', 4: 'purple', 5: 'orange', 6: 'black',
                    7: 'pink', 8:'brown', 9:'DarkBlue'}

    # 设置类名
    cluster_names = {0: u'类0', 1: u'类1', 2: u'类2', 3: u'类3', 4: u'类4', 5: u'类5', 6: u'类6', 7: u'类7',
                         8: u'类8', 9: u'类9'}

    df = pd.DataFrame(dict(x=xs, y=ys, label=y_pred, title=corpus))
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(8, 5))  # set size
    ax.margins(0.02)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=cluster_names[name],
                    color=cluster_colors[name], mec='none')
    plt.show()
    plt.savefig('res.png')

read_data()
classify('clean.txt', 20)
a = np.load('y_pred.npy')

for i in range(len(a)//10):
    print(a[i*10:(i+1)*10])
