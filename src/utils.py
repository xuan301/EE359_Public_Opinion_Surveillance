import jieba
import re

# jieba.load_userdict("user_dict.txt")
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
    if(line != ""):
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