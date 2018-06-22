from __future__ import division
import numpy as np
import random
import sys
import os

def norm(a):
	temp = list()
	mn, mx = min(a), max(a)

	for i in range(0,len(a),1):
		x = (a[i]-mn)/(mx-mn)
		temp.append(x)
	return temp

file_name2 = sys.argv[1]
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

name = file_name[:file_name.rfind('.')]
seg = os.path.join('score', name) + '.seg.score'
sys = os.path.join('score', name) + '.sys.score'

f1 = open(seg,'w+')
f2 = open(sys,'w+')

s = 0
for i in range(0,len(lines),1):

	sent, lp, sname, e1, e2, cs, s1, s2 = lines[i].split('\t')
	e1, e2, cs, x = e1_list[i], e2_list[i], cos_list[i], plist[i]

	score_line = [str(i), s1, s2, str(e1), str(e2), str(lp), str(cs), str(x)]

	curr_line = [name,lp,"newstest2014",sname,str(i%3003+1),str(x)]
	f1.write('\t'.join(curr_line[0:]) + '\n')
	s += x
	if (i+1) % 3003 == 0:
		avg = s/3003
		curr_line2 = [name,lp,"newstest2014",sname,str(avg)]
		f2.write('\t'.join(curr_line2[0:]) + '\n')
		s = 0

f1.close()
f2.close()