#!/usr/bin/env python
import sys
import os


def main():

	#ip, gs = sys.argv[1], sys.argv[2]

	file_name = sys.argv[1]

	f1 = open(file_name).read().strip().split('\n')

	s1, s2, label = 's1.train', 's2.train', 'labels.train'

	fs1, fs2, flabel = open(s1,'a+'), open(s2,'a+'), open(label,'a+')

	print len(f1)

	for i in range(1,len(f1),1):

		line = f1[i].split('\t')
		sent1, sent2, lb = line[1], line[2], line[4]
		fs1.write(sent1 + '\n')
		fs2.write(sent2 + '\n')
		flabel.write(lb + '\n')

	fs1.close()
	fs2.close()
	flabel.close()

if __name__ == "__main__" : 
	main()
