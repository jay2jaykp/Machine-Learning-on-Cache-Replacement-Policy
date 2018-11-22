#from tqdm import tqdm_notebook as tqdm
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
from sklearn.preprocessing import normalize
from collections import deque
from collections import defaultdict
import numpy as np

num_params = 2

def getRelativeTS(Cindex,df):
	TS = np.array([])
	for e in Cindex:
		TS = np.append(TS,df.iloc[int(e)]['timestamp'])
	norm1 = TS / np.linalg.norm(TS)
	return norm1

def getRelativePos(Cindex,df):
	BN = np.array([])
	for e in Cindex:
		BN = np.append(BN,df.iloc[int(e)]['bno'])
	norm1 = BN / np.linalg.norm(BN)
	#print (norm1)
	return norm1

def getRelativeFrequency(Cindex,df):
	return 0

def getRelativeRecency(Cindex,df):
	return 0

def getY(evictList, CacheList):
	Y = np.empty([])
	for e in CacheList:
		if e in evictList:
			Y = np.append(Y,1)
		else:
			Y = np.append(Y,0)
	Y = np.delete(Y,0)
	return Y

# add more parameters and increase num_params above

def getX(evictIndex, df):
	x_t = getRelativeTS(evictIndex,df)
	x_p = getRelativePos(evictIndex,df)
	#	x_f = getRelativeFrequency(evictIndex,df)
	#	x_r = getRelativeRecency(evictIndex,df)
	#	X = np.vstack((X,np.array([x_t,x_p,x_f,x_r])))
		# = np.vstack((X,np.array([x_t,x_p])))
	X = (np.array((x_t,x_p))).T
	return X
