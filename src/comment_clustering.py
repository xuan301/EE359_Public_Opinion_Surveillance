from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud,ImageColorGenerator
import random
from PIL import Image

random.seed(999)

case = "case5"

cluster_num = 20

comment = []
txt = open('output/{}_comment.txt'.format(case), 'r', encoding="utf-8")
for s in txt:
    comment.append(s[:-1])

corpus = []
txt = open('output/{}_corpus.txt'.format(case), 'r', encoding="utf-8")
for s in txt:
    corpus.append(s[:-1])
print("comment number: ", len(corpus))
print("computing TF-IDF...")
vectorizer = CountVectorizer(max_features=30000)
# 该类会统计每个词语的tf-idf权值
tf_idf_transformer = TfidfTransformer()
# 将文本转为词频矩阵并计算tf-idf
tfidf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(corpus))
# 获取词袋模型中的所有词语
tfidf_matrix = tfidf.toarray()
print(tfidf_matrix)
np.save('output/{}_tfidf_matrix.npy'.format(case), tfidf_matrix)
print('Starting Clustering...')


# Z = linkage(tfidf_matrix, 'ward')
# plt.figure(figsize=(50, 10))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
# # dendrogram(Z, truncate_mode='lastp', 
# #             p=12, show_leaf_counts=False, 
# #             leaf_rotation=90., leaf_font_size=12., 
# #             show_contracted=True)
# plt.axis('off')
# plt.savefig("level_tree.png")
# plt.show()

# clusters = fcluster(Z, 6, criterion='maxclust')

model = KMeans(n_clusters=cluster_num, random_state=123)
model.fit(tfidf_matrix)
clusters = model.labels_

corpus_clusters = {}
for i in range(len(clusters)):
    label = str(clusters[i])
    if label not in corpus_clusters:
        corpus_clusters[label] = [corpus[i]]
    else:
        corpus_clusters[label].append(corpus[i])

comment_clusters = {}
for i in range(len(clusters)):
    label = str(clusters[i])
    if label not in comment_clusters:
        comment_clusters[label] = [comment[i]]
    else:
        comment_clusters[label].append(comment[i])

with open("output/{}_clustered_comment.txt".format(case), 'w', encoding='utf-8') as f:
    for i in comment_clusters:
        for c in comment_clusters[i]:
            f.writelines([i,"\t", c, "\n"])

with open("output/{}_clustered_corpus.txt".format(case), 'w', encoding='utf-8') as f:
    for i in corpus_clusters:
        for c in corpus_clusters[i]:
            f.writelines([i,"\t", c, "\n"])

cloud_img = np.array(Image.open("cloud.jpg"))


for i in corpus_clusters:
    text = ""
    for c in corpus_clusters[i]:
        text = text + " " + c
    wc = WordCloud(width=1400, height=2200,
			background_color='white',
	        mode='RGB',
			max_words=10,
            mask=cloud_img,
			stopwords="dict/stopwords.txt",
            font_path='dict/simsun.ttf',
			max_font_size=500,
			relative_scaling=0.6,
			random_state=50,
			scale=2
			).generate(text)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()