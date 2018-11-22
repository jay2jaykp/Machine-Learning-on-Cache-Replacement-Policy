'''
get the furthest accessed block. Scans OPT dictionary and selects the maximum positioned element
'''

#!/usr/bin/python3

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
import tqdm


import helper as h
from collections import defaultdict
from functools import partial
import numpy as np

def getFurthestAccessBlock(C, OPT):
	maxAccessPosition = -1
	maxAccessBlock = -1
	for cached_block in C:
		if len(OPT[cached_block]) is 0:
		#print ( "Not Acccessing block anymore " + str(cached_block))
			return cached_block            
	for cached_block in C:  
		if OPT[cached_block][0] > maxAccessPosition:
			maxAccessPosition = OPT[cached_block][0]
			maxAccessBlock = cached_block
	return maxAccessBlock

def getFurthestAccessBlocks(C, OPT, evict_ratio):
	evict_list = np.array([]);
	for i in np.arange(0,(len(C) * evict_ratio / 100)):
		fblk = getFurthestAccessBlock(C, OPT)
		evict_list = np.append(evict_list, fblk)
		idx = np.where(C==fblk)
		C = np.delete(C,idx)
	return evict_list
			

def build_dict(blocktrace):
	OPT = defaultdict(partial(np.ndarray,0))

	for i, block in enumerate(blocktrace):
		OPT[block] = np.append(OPT[block], i)

	return OPT
	

def belady_opt(blocktrace,sequences,cache_size, df):
	OPT = defaultdict(partial(np.ndarray,0))

	for i, block in enumerate(tqdm.tqdm(blocktrace, desc="OPT: building index")):
		OPT[block] = np.append(OPT[block], i)    

	print ("created OPT dictionary")    

	hit, miss = 0, 0

	C = deque() # Cache
	C2 = deque()
	d = defaultdict(deque)
	d_ftime = set()
	d_timestamp = {}
	d_label = {}

	for k,block in enumerate(tqdm.tqdm(blocktrace, desc="OPT", leave=False)):

		if block in C:
			OPT[block] = np.delete(OPT[block],0)
			hit+=1
		else:
			miss+=1
			if len(C) == cache_size:
				fblock = getFurthestAccessBlock(C, OPT)
				assert(fblock != -1)
				d[fblock] = deque(zip(C,C2)) 
				d_timestamp[fblock] = h.getRelativeTS(list(C2), df) #Returning Time Stemp
				d_label[fblock] = C.index(fblock)
				C2.remove(C2[C.index(fblock)])
				C.remove(fblock)
				#C2.remove()
				d_ftime.add(sequences[k])
				return d,d_timestamp, d_label
			C.append(block)
			C2.append(k)
			d_ftime.add(sequences[k])
			#OPT[block] = OPT[block][1:]
			#print(OPT)
			OPT[block] = np.delete(OPT[block],[0])

	#print ("hit count" + str(hit_count))
	#print ("miss count" + str(miss_count))
	hitrate = hit / (hit + miss)
	print(hitrate)
	return d,d_timestamp,d_label
