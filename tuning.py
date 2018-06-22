from __future__ import division
import numpy as np
import random
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd


def norm(a):
	temp = list()
	mn, mx = min(a), max(a)

	#print mn, mx, a[0], a.index(mn), a.index(mx)

	for i in range(0,len(a),1):
		x = (a[i]-mn)/(mx-mn)
		temp.append(x)
	return temp


def train_model():
	train_data=pd.read_csv("tuning_train.desc", sep = '\t',header=None)

	trainx=train_data.ix[:,1:3]
	trainy=train_data.ix[:,6:6]

	features = trainx.values
	target = trainy.values

	neighbors = 10
	wts = 'distance'  # can be 'uniform' or 'distance'


	regr = LinearRegression()
	regr_model = regr.fit(features, target.ravel())

	return regr_model

def test_model(regr_model, file_name2):
	
	file_name = os.path.basename(file_name2)
	lines = open(file_name2).read().strip().split('\n')

	cos_list = [float(temp.split('\t')[5]) for temp in lines]
	e1_list = [float(temp.split('\t')[3]) for temp in lines]
	e2_list = [float(temp.split('\t')[4]) for temp in lines]
	plist =  [float(temp.split('\t')[3])*float(temp.split('\t')[4]) for temp in lines]

	cos_list = norm(cos_list)
	e1_list = norm(e1_list)
	e2_list = norm(e2_list)
	plist = norm(plist)

	validate2 = pd.read_csv(file_name2, sep='\t', header=	None)
	print (validate2.shape)
	testx = validate2.ix[:, 3:5]
	print (testx)
	
	test_features = testx.values
	prediction_regr = regr_model.predict(test_features)


	name = file_name[:file_name.rfind('.')]
	seg = os.path.join('tuned_score', name) + '.seg.score'
	sys = os.path.join('tuned_score', name) + '.sys.score'

	f1 = open(seg,'w+')
	f2 = open(sys,'w+')

	s = 0
	for i in range(0,len(lines),1):

		sent, lp, sname, e1, e2, cs, s1, s2 = lines[i].split('\t')
		e1, e2, cs, pi = e1_list[i], e2_list[i], cos_list[i], plist[i]
		
		curr_line = [name,lp,"newstest2014",sname,str(i%3003+1),str(prediction_regr[i])]
		f1.write('\t'.join(curr_line[0:]) + '\n')
		s += prediction_regr[i]
		if (i+1) % 3003 == 0:
			avg = s/3003
			curr_line2 = [name,lp,"newstest2014",sname,str(avg)]
			f2.write('\t'.join(curr_line2[0:]) + '\n')
			s = 0

	f1.close()
	f2.close()


def main():
	regr_model = train_model()
	file_name2 = sys.argv[1]
	test_model(regr_model,file_name2)

if __name__ == '__main__':
    main()