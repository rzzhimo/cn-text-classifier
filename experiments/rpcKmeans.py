#!/usr/bin/python
# -*- coding:utf-8 -*-
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
import time
import numpy as np
from tqdm import tqdm
from importlib import import_module
import argparse
import logging
import urllib
import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.web import RequestHandler
import py_eureka_client.eureka_client as eureka_client
from tornado.options import define, options
import json
from tools.rpcCutWord import *
define("port", default=3334, help="run on the given port", type=int)
textList1 = ['新冠肺炎遗体解剖揭开『新冠肺炎的秘密-廖晋堂医师(中文字幕)', '为什么说新冠病毒是人类历史上最难对付的病毒之一？', '“美国新冠肺炎患者跳海致海盐污染”是谣言!(转载)', '印度首都新德里监狱系统累计有221人感染新冠肺炎', '印度首都新德里监狱系统累计有221人感染新冠肺炎', '强证据表明羟氯喹预防新冠不比安慰剂更有效', '《自然》子刊:新冠病毒谱系可能已在蝙蝠中传播数十年', '高考防疫家长怎么做？国家卫健委发布10条关键提示', '重磅!北大合作团队新冠强效药研发有新进展(转载)', '关于制定新冠病毒核酸检测收费标准的通知-安顺市发展和改革委员会（安顺市粮食局）', '台湾新增5例新冠肺炎确诊病例-新华网']

def prepare(textList):
    wordCut = WordCut()
    print(settings.STATIC_DIR + "dict.txt")
    wordCut.addDictionary(settings.STATIC_DIR + "dict.txt")
    sentences = wordCut.seg_file(textList, True)

    print(len(sentences))
    return sentences
logging.basicConfig(level=logging.INFO)
def listToJson(lst):
    keys = [str(x) for x in np.arange(len(lst))]
    list_json = dict(zip(keys, lst))
    str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
    return str_json

def Kmeans(textList):
    sentences = prepare(textList)
    # 词频矩阵 Frequency Matrix Of Words
    vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    transformer = TfidfTransformer()
    # Fit Raw Documents
    freq_words_matrix = vertorizer.fit_transform(sentences)
    # Get Words Of Bag
    words = vertorizer.get_feature_names()
    tfidf = transformer.fit_transform(freq_words_matrix)
    weight = freq_words_matrix.toarray()

    # #找与第一句相似度高的sentences
    # cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
    # related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    # print(related_docs_indices)
    # print(cosine_similarities[related_docs_indices])

    pca = PCA(n_components=5)
    trainingData = pca.fit_transform(weight)

    numOfClass = 10
    clf = KMeans(n_clusters=numOfClass, max_iter=10000,  tol=1e-6, random_state=5)
    result = clf.fit(trainingData)
    source = list(clf.predict(trainingData))
    label = clf.labels_
    center = clf.cluster_centers_

    # # 找与第一个中心相似度高的sentences
    # cosine_similarities = linear_kernel(center[0:1], trainingData).flatten()
    # related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    # print(related_docs_indices)
    # print(cosine_similarities[related_docs_indices])

    print(label)
    r1 = pd.Series(clf.labels_).value_counts()  # 统计各个类别的数目

    r2 = pd.DataFrame(clf.cluster_centers_)  # 找出聚类中心

    r = pd.concat([r2, r1], axis=1)  # 横向连接(0是纵向), 得到聚类中心对应的类别下的数目
    res0Series = pd.Series(clf.labels_)
    res0 = res0Series[res0Series.values == 1]
    # sentencesSeries = pd.Series(sentences)
    # print("类别为1的数据\n", (sentencesSeries.iloc[res0.index]))
    print(r)
    return label

class IndexHandler(tornado.web.RequestHandler):

    def post(self, *args, **kwargs):
        j = json.loads(self.request.body.decode('utf-8'))
        # print(j["textList"])
        textList = j["textList"]
        predict_all = []
        if(len(textList)<=10):
            for i in range(len(textList)):
                predict_all.append(str(i))
        else:
            result = Kmeans(textList)
            for i in range(len(result)):
                predict_all.append(str(result[i]))
        result1 = listToJson(predict_all)
        print(''.join(result1))
        self.write(''.join(result1))
def main():
    tornado.options.parse_command_line()
    # 注册eureka服务
    eureka_client.init(eureka_server="http://localhost:9000/eureka/",
                                       app_name="clustering-service",
                                       instance_port=3334)
    app = tornado.web.Application(handlers=[(r"/kmeans", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    Kmeans(textList1)
    main()