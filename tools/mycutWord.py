# -*- coding: utf-8 -*-
import jieba
import settings
import json
import csv
import codecs
import pandas as pd
import re

#stopwords_path = os.path.normpath(os.path.dirname(__file__)) + "/stopwords.txt"
stopwords_path = settings.STATIC_DIR + "/stopwords.txt"


class WordCut(object):
    def __init__(self, stopwords_path=stopwords_path):
        """
        :stopwords_path: 停用词文件路径

        """
        self.stopwords_path = stopwords_path

    def addDictionary(self, dict_list):
        """
        添加用户自定义字典列表
        """
        map(lambda x: jieba.load_userdict(x), dict_list)

    def seg_sentence(self, sentence, stopwords_path=None):
        """
        对句子进行分词
        """
        # print "now token sentence..."
        if stopwords_path is None:
            stopwords_path = self.stopwords_path

        def stopwordslist(filepath):
            """
            创建停用词list ,闭包
            """
            stopwords = [line.encode('utf-8').decode('utf-8').strip() for line in open(filepath, 'r').readlines()]
            return stopwords

        sentence_seged = jieba.cut(sentence.strip())
        stopwords = stopwordslist(stopwords_path)  # 这里加载停用词的路径
        outstr = ''  # 返回值是字符串
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t' and word!='\n':
                    outstr += word
                    outstr += " "
        return outstr

    def seg_file(self, path, show=True, write=False):
        """
        对文本进行分词
        """
        print("now token file...")
        # if write is True:
        #     write_path = '/'.join(path.split('/')[:-1]) + '/y_cut_data.csv'
        #     w = open(write_path, 'wb+')
        lines_list = []
        with open(path, 'r') as p:
            reader = csv.reader(p)
            column1 = [row[1] for row in reader]
            index = 0;
            for line in column1:
                if(index<1):
                    index+=1
                    continue
                line_seg = self.seg_sentence(line)
                # lines_list.append(line_seg)

                linec = {'index':index,
                        'content':line_seg,
                        }
                lines_list.append(linec)
                if show is True:
                    print(index)
                    print(line_seg)
                index=index+1
        if write is True:
            write_path = '/'.join(path.split('/')[:-2]) + '/y_cut_data.csv'
            data = pd.DataFrame(lines_list)
            number = 1
            # 写入csv文件,'a+'是追加模式
            try:
                if number == 1:
                    csv_headers = ['index', 'content']
                    data.to_csv(write_path, header=csv_headers, index=False, mode='w+', encoding='utf-8')
                else:
                    data.to_csv(write_path, header=False, index=False, mode='w+', encoding='utf-8')
                    number = number + 1
            except UnicodeEncodeError:
                print("编码错误, 该数据无法写到文件中, 直接忽略该数据")

    def my_file(self,path,show=True,write=False):
        """
        对舆情文本进行转化
        """
        print("now load file...")
        if write is True:
            write_path1 = '/'.join(path.split('/')[:-1]) + '/y_data_t.csv'
            write_path2 = '/'.join(path.split('/')[:-1]) + '/y_data.csv'
            w1 = open(write_path1, 'wb+')
            w2 = open(write_path2, 'wb+')

        with open(path, "r") as f:
            json_dict = json.loads(f.read())
            #i = 1
            print(len(json_dict["response"]["docs"]))
            title_list = []
            content_list = []
            for doc in json_dict["response"]["docs"]:
                #if(i>10):break
                title = doc.pop("title")
                content = doc.pop("content").replace('\n', '')
                content = re.sub(r'\s{2,}', '', content)
                if show is True:
                    print(content)
                if write is True:
                    w1.write(title.encode('utf-8'))
                    w1.write('\n'.encode('utf-8'))
                    w2.write(content.encode('utf-8'))
                    w2.write('\n'.encode('utf-8'))
                #i=i+1
        if write is True:
            w1.close()
            w2.close()
        count = 0
        for index, line in enumerate(open(write_path2, 'r')):
            count += 1
        print(count)
        count = 0
        for index, line in enumerate(open(write_path1, 'r')):
            count += 1
        print(count)
    def my_file2(self,path,show = False,write=False):
        """
        对舆情文本进行转化
        """
        print("now load file...")
        title_list = []
        content_list = []
        with open(path, 'r') as f:
            json_dict = json.loads(f.read())
            index = 1
            for doc in json_dict["response"]["docs"]:
                #if(i>10):break
                title = doc.pop("title")
                content = doc.pop("content").replace('\n', '')
                content = re.sub(r'\s{2,}', '', content)
                if (len(content) <= 10):
                    print(content)
                    continue
                linet = {'index': index,
                         'content': title,
                         }
                linec = {'index': index,
                         'content': content,
                         }
                title_list.append(linet)
                content_list.append(linec)
                index = index+1
        if write is True:
            write_path1 = '/'.join(path.split('/')[:-1]) + '/y_data_t.csv'
            write_path2 = '/'.join(path.split('/')[:-1]) + '/y_data.csv'
            data1 = pd.DataFrame(title_list)
            data2 = pd.DataFrame(content_list)
            number = 1
            # 写入csv文件,'a+'是追加模式
            try:
                if number == 1:
                    csv_headers = ['index', 'content']
                    data1.to_csv(write_path1, header=csv_headers, index=False, mode='w+', encoding='utf-8')
                    data2.to_csv(write_path2, header=csv_headers, index=False, mode='w+', encoding='utf-8')
                else:
                    data1.to_csv(write_path1, header=False, index=False, mode='w+', encoding='utf-8')
                    data2.to_csv(write_path1, header=False, index=False, mode='w+', encoding='utf-8')
                    number = number + 1
            except UnicodeEncodeError:
                print("编码错误, 该数据无法写到文件中, 直接忽略该数据")
wordCut = WordCut()
print(settings.STATIC_DIR+"dict.txt")
wordCut.addDictionary(settings.STATIC_DIR+"dict.txt")
wordCut.seg_file("/Users/apple/Mac拓展/大四考研找工作/自动聚类/cn-text-classifier/srcData/original/y_data.csv",False,True),
#wordCut.my_file2("/Users/apple/Mac拓展/大四考研找工作/自动聚类/cn-text-classifier/srcData/original/solr.json",False,True)