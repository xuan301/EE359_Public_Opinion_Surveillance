import pandas as pd
import utils

case = "case5"

if case == "case1":
    file = pd.read_excel('dataset/case1_南京警方通报男子驾车撞人并持刀捅人.xlsx')
elif case == "case2":
    file = pd.read_excel('dataset/case2_南京胖哥.xlsx')
elif case == "case3":
    file = pd.read_excel('dataset/case3_泡泡玛特致歉.xlsx')
elif case == "case4":
    file = pd.read_excel('dataset/case4_日本政府正式决定福岛核废水排海.xlsx')
elif case == "case5":
    file = pd.read_excel('dataset/case5_三胎政策来了.xlsx')
elif case == "case6":
    file = pd.read_excel('dataset/case6_任豪.xlsx')

data = file.values
stopword_list = utils.stopwordslist('dict/stopwords.txt')

comment_list = []
with open('output/{}_comment.txt'.format(case), 'w', encoding='utf-8') as f:
    for row in data:
        comment = ""
        if not isinstance(row[15], float): # this is a second-level comment
            comment = row[15][row[15].find("：")+1:] # delete the commenter
            comment = comment[comment.find(":")+1:] # delete the reply info
            if comment == "" or comment[0] == "@":
                comment == ""
        else: # it is a first-level comment
            print(row[11])
            comment = row[11]
            if isinstance(comment, float) or comment == "" or comment[0] == "@":
                comment == ""

        if not isinstance(comment, str) or comment == "":
            continue
        if comment[0] == "@":
            continue
        comment_list.append(comment)
        f.writelines(comment+"\n")


with open('output/{}_corpus.txt'.format(case), 'w', encoding='utf-8') as f:
    for comment in comment_list:
        comment = utils.clear_line(comment)
        comment = utils.sent2word(comment, stopword_list)
        f.writelines(comment+"\n")

