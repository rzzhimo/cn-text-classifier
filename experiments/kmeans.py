# -*- coding: utf-8 -*-
"""
K-means-Single-Test
"""

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel

from tools.preprocess import *
from tools.visualizer import plot_result
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from tools.labelText import LabelText
import settings
import time
import pandas as pd
from sklearn import metrics


"""
loading source
载入资源
文件详情参照本文件夹README
"""
print('------Loading Source...')
cut_path = settings.SOURCE_DATA + 'y_cut_data.csv'
ori_path = settings.SOURCE_DATA + 'original/y_data.csv'
# sentences = loading_source(file_name=ori_path)
sentences = []
# content_lines = loading_source(file_name=ori_path)
# ori_path = settings.SOURCE_DATA + 'cut_data.csv'
sentences = loading_source(file_name=cut_path)
print(len(sentences))

# start = time.time()
# cut_source(content_lines, sentences, write=True)
# end = time.time()
# print('------- cutting cost', end - start)


"""
Vertorizer
向量化
"""
print('------Vertorizer...')
start = time.time()

# 词频矩阵 Frequency Matrix Of Words
vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
transformer = TfidfTransformer()
# Fit Raw Documents
freq_words_matrix = vertorizer.fit_transform(sentences)
# Get Words Of Bag
words = vertorizer.get_feature_names()
tfidf = transformer.fit_transform(freq_words_matrix)

weight = freq_words_matrix.toarray()

end = time.time()

# print ('vocabulary list:\n')
# for key,value in vertorizer.vocabulary_.items():
#     print (key,value)

print("Shape: Documents(Class) / Words")
print(weight.shape)

# print("------ IFIDF词频矩阵:\n")
# print(weight)

# for i in range(10):
# # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，
# #第二个for遍历某一类文本下的词语权重
#     print (u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
#     for j in range(len(words)):
#         if(weight[i][j]>0):
#             print (words[j], weight[i][j])#第i个文本中，第j个词的tfidf值


print('------ vectorizer cost', end-start)


"""
Dimension Reduction
降维
"""
start = time.time()
pca = PCA(n_components=5)
trainingData = pca.fit_transform(weight)
print(pca.explained_variance_ratio_)
# svd = TruncatedSVD(n_components=10, n_iter=10, random_state=42)
# trainingData = svd.fit_transform(weight)
end = time.time()
print('------ Dimension Reduction', end-start)

"""
Compute K-Means
"""

numOfClass: int = 10

start = time.time()
clf = KMeans(n_clusters=10, max_iter=10000, init="k-means++", tol=1e-6, random_state=0)

result = clf.fit(trainingData)
source = list(clf.predict(trainingData))
end = time.time()

label = clf.labels_
center = clf.cluster_centers_

labelAndText = LabelText(label, ori_path)

print('------ Compute K-Means', end-start)
labelAndText.sortByLabel(show=False, write=True)
#labelAndText.arrangeLabelText(False,True)

# 找与每个中心相似度高的sentences
for i in range(10):
    cosine_similarities = linear_kernel(center[i:i+1], trainingData).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    print(related_docs_indices)
    print(cosine_similarities[related_docs_indices])
    if(cosine_similarities[related_docs_indices][0]>0.9):
        print(sentences[related_docs_indices[0]])



"""
Result
生成各个指标并写入文件
"""
content = pd.read_csv(settings.SOURCE_DATA + 'labeled_data.csv')
labels_true = content.flag.to_list()


# ars = metrics.adjusted_rand_score(labels_true, label)
# print("adjusted_rand_score: ", ars)
#
# fmi = metrics.fowlkes_mallows_score(labels_true, label)
# print("FMI: ", fmi)

silhouette = metrics.silhouette_score(trainingData, label)
print("silhouette: ", silhouette)

CHI = metrics.calinski_harabasz_score(trainingData, label)
print("CHI: ", CHI)

# with open(settings.DST_DATA+time.strftime('KM'+"%Y-%m-%d %H-%M-%S", time.localtime())+'result.txt', 'w') as w:
#     w.write("------K-Means Experiment-------\n")
#     # w.write("adjusted_rand_score: %f\n" % ars)
#     # w.write("FMI: %f\n" % fmi)
#     w.write("Silhouette: %f\n " % silhouette)
#     w.write("CHI: %f\n" % CHI)
#     w.write("------End------")

plot_result(trainingData, source, numOfClass)


