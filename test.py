from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math


def cal_entropy(text):
    h = 0.0
    sum = 0
    letter = [0] * 26
    text = text.lower()
    for char in text:
        if char.isalpha():
            letter[ord(char) - ord('a')] += 1
            sum += 1
    for i in range(26):
        p = 1.0 * letter[i] / sum
        if p > 0:
            h += -(p * math.log(p, 2))
    return h


# 定义notdga为0,dga为1

def deal_with_traindata(filename):
    Domainlist = []
    LableList = []
    with open(filename, 'r') as f:

        for line in f:
            tokens = line.split(",")
            length = len(tokens[0])
            en = cal_entropy(tokens[0])
            seg = len(tokens[0].split("."))
            num = 0
            for char in tokens[0]:
                if char.isdigit():
                    num += 1

            if tokens[1] == 'notdga\n':
                lable = 0
            else:
                lable = 1

            Domainlist.append([length, num, en, seg])
            LableList.append(lable)
    return Domainlist, LableList


def deal_with_testdata(filename):
    Testlist = []
    Domainnamelist=[]
    with open(filename, 'r') as f:
        for sample in f:
            Domainnamelist.append(sample.rstrip('\n'))
            length = len(sample)
            en = cal_entropy(sample)
            seg = len(sample.split("."))
            num = 0
            for char in sample:
                if char.isdigit():
                    num += 1
            Testlist.append([length, num, en, seg])

    return Testlist,Domainnamelist


featureMatrix, labelList = deal_with_traindata("train.txt")

clf = RandomForestClassifier(random_state=0)
clf.fit(featureMatrix, labelList)

testMatrix ,Domainnamelist= deal_with_testdata("test.txt")
resultlist = []

with open("result.txt", "w+", newline='') as f:
    for i in range(len(testMatrix)):
        t = clf.predict([testMatrix[i]])
        if t == 0:
            resultlist.append(Domainnamelist[i] + ',' + 'notdga')
        elif t == 1:
            resultlist.append(Domainnamelist[i] + ',' + 'dga')
    for result in resultlist:
        f.write(result + '\n')
