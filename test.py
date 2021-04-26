from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

class Domain:
	def __init__(self,  _name, _label, _length, _num, _entropy):
		self.name = _name
		self.label = _label
		self.length = _length
		self.num = _num
		self.entropy = _entropy
	def returnData(self):
		return [self.length, self.num, self.entropy]
	def returnLabel(self):
		if self.label == "notdga":
			return 0
		else:
			return 1

def calculateNum(str):
        num = 0
	for i in str:
		if i.isdigit():
			num += 1
	return num

def calculateEntropy(str):
	#统计字符数目
	result = {}
	for i in str:
		result[i] = str.count(i)
	sum = len(str)
	#计算熵值
	entropy = 0
	for j in result:
		entropy = entropy - float(result[j] / sum) * math.log(float(result[j] / sum), 2)
	return entropy

def initData(filename, domainlist):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line == "":
				continue
			tokens = line.split(",")
			name = tokens[0]
			if len(tokens) > 1
				label = tokens[1]
			else
				label = "unknown"
			length = len(name)
			num = calculateNum(name)
			entropy = calculateEntropy(name)
			domainlist.append(Domain(name, label, length, num, entropy))
			
def main():
	#读取训练文件
	domainlist1 = []
	initData("train", domainlist1)
	featureMatrix = []
	labelList = []
	for item1 in domainlist1:
		featureMatrix.append(item1.returnData())
		labelList.append(item1.returnLabel())
	#训练
	clf = RandomForestClassifier(random_state = 0)
	clf.fit(featureMatrix, labelList)
	#读取测试文件
	domainlist2 = []
	initData("test", domainlist2)
	with open("result", 'w') as f:
		for item2 in domainlist2
			f.write(item2.name)
			f.write(", ")
			f.write(clf.predict([item2.returnData]))
			f.write("\n")
	
if __name__ == '__main__':
	main()
