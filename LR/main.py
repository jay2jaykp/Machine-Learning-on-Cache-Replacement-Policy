# -*- coding: utf-8 -*-

import helper as h
import belady as b
import pandas as pd
from tqdm import tqdm
import numpy as np

from functools import partial
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure 
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import deque
from collections import defaultdict
import sys

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.metrics import confusion_matrix


import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

cache_size = 1000
evict_block_ratio = 10
skip_interval = 0 # skip only cache_size elements by default.
	# skip more to get more heterogenous data

df100 = pd.read_csv('../data/ofile.csv', sep=',')
df100.columns = ['no','timestamp','pid','pname','bno', 'bsize', 'op', 'dvmajor', 'dvminor', 'blockhash']
#df100.columns = ['no']

blktrace = df100['bno'].tolist()
blocktrace = np.array([int(x) for x in blktrace])

#seq = df['no'].tolist()
seq = df100['no'].tolist()
sequences = np.array([int(x) for x in seq])

OPT = b.build_dict(blocktrace)

# create a slice of unique blocks of size cache_size.
# get evicted blocks according to OPT algorithm for these slices.
# keep skipping already considered blocks, build new slice and repeat.

print( "unique entries " + str(len(set(blocktrace))))

X = np.zeros((h.num_params,))
Y = np.zeros((1,))

done = False

for cacheRange in tqdm(np.arange(0,len(blocktrace), cache_size)):
	C = np.array([])
	Cindex = np.array([]) # stores indices of blocks in cache
	Y_current = np.array([])
	X_current = np.array([])

	# number of blocks that should be skipped
	skip_count = 0

	for e in np.arange(0, cache_size):
		# skip repeated entries
		while(((e+skip_count) < (len(blocktrace) -1 )) and blocktrace[e+skip_count] in C):
			OPT[blocktrace[e+skip_count]] = np.delete(OPT[blocktrace[e+skip_count]],0)
			skip_count+=1
		if((len(blocktrace) -1) < e+skip_count):
			done=True
			break
		C = np.append(C,blocktrace[e+skip_count])
		Cindex = np.append(Cindex,sequences[e+skip_count])

	if done is True:
		break

	# remove cached elements from OPT and increment blocktrace
	for e in C:
		OPT[e] = np.delete(OPT[e],0)
	blocktrace = blocktrace[cache_size+skip_interval + skip_count:]
	sequences = sequences[cache_size+skip_interval + skip_count:]

	# call eviction procedure.
	evict_list = b.getFurthestAccessBlocks(C, OPT, evict_block_ratio)
	Y_current = h.getY(evict_list, C)
	X_current = h.getX(Cindex, df100)

	Y = np.append(Y,Y_current)
	X = np.vstack((X,X_current))

#remove zeros line required for vstack

X = X[1:]
Y = Y[1:]

#Train-Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size=0.3, random_state=0)

#Fitting Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

confusion_matrix = confusion_matrix(Y_test,Y_pred)
print (confusion_matrix)
