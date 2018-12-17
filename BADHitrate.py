#!/usr/bin/env python
# coding: utf-8

# In[458]:

from tqdm import tqdm as tqdm 
import numpy as np
from collections import deque, defaultdict
import timeit
import pandas as pd
import random
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import sys

# dummy maxmimum position variable. assign the position of blocks that 
# will never get accessed a value greater than this value. this way OPT
# can be fooled to think that the block will be accessed but at a position
# far-far-away in time.

maxpos = 1000000000000

num_params = 3
sampling_freq = 1000 # number of samples skipped
cache_size = 1000    # default cache size
eviction = int(0.1 * cache_size)  # number of blocks evicted
filename = "ikki-110108-112108.1.blkparse"
#filename = "cheetah.1000"

df = pd.read_csv(filename, sep=' ',header = None)
df.columns = ['timestamp','pid','pname','blockNo', 'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']
df.head()

# In[460]:

blocktrace = df['blockNo'].tolist()

timestamp = df['timestamp'].tolist()

le = preprocessing.LabelEncoder()

le.fit(df['pid'].tolist())

pid = le.transform(df['pid'].tolist())


# In[466]:


#LRU(blocktrace, 500)


# In[467]:


def LFU(blocktrace, frame):
    
    cache = set()
    cache_frequency = defaultdict(int)
    frequency = defaultdict(int)
    
    hit, miss = 0, 0
    
    for block in tqdm(blocktrace):
        frequency[block] += 1
        
        if block in cache:
            hit += 1
            cache_frequency[block] += 1
        
        elif len(cache) < frame:
            cache.add(block)
            cache_frequency[block] += 1
            miss += 1

        else:
            e, f = min(cache_frequency.items(), key=lambda a: a[1])
            cache_frequency.pop(e)
            cache.remove(e)
            cache.add(block)
            cache_frequency[block] = frequency[block]
            miss += 1
    
    hitrate = hit / ( hit + miss )
    return hitrate

'''
    given C, use LFUDict to find eviction number of blocks from the Cache
    compare it with Y_OPT and store number of places the two differ
'''
lruCorrect = 0
lruIncorrect = 0

def lruPredict(C,LRUQ,Y_OPT):
    global lruCorrect, lruIncorrect
    Y_current = []
    KV = defaultdict(int)
    for i in range(len(LRUQ)):
        KV[LRUQ[i]] = len(LRUQ) - i
    KV_sorted = Counter(KV)
    evict_dict = dict(KV_sorted.most_common(eviction))
    for e in C:
        if e in evict_dict:
            Y_current.append(1)
        else:
            Y_current.append(0)
    for i in range(len(Y_current)):
        if Y_current[i] is Y_OPT[i]:
            lruCorrect+=1
        else:
            lruIncorrect+=1
    return Y_current

# returns sequence of blocks in prioirty order

def Y_getBlockSeq(Y_pred_prob):
    x = []
    for i in range(len(Y_pred_prob)):
        x.append(Y_pred_prob[i][0])
    x = np.array(x)
    idx = np.argsort(x)
    idx = idx[:eviction]
    return idx

def Y_getMinPredict(Y_pred_prob):
    x = []
    for i in range(len(Y_pred_prob)):
        x.append(Y_pred_prob[i][0])
    x = np.array(x)
    idx = np.argpartition(x, eviction)
    
    Y_pred = np.zeros(len(Y_pred_prob), dtype=int)
    for i in range(eviction):
        Y_pred[idx[i]] = 1
    assert(Counter(Y_pred)[1] == eviction)
    return Y_pred

'''
    given C, use LFUDict to find eviction number of blocks from the Cache
    compare it with Y_OPT and store number of places the two differ

    The number of correct and incorrect predictions with respect to OPT.
'''

lfuCorrect = 0
lfuIncorrect = 0

def lfuPredict(C,LFUDict,Y_OPT):
    global lfuCorrect, lfuIncorrect
    Y_current = []
    KV = defaultdict()
    for e in C:
        KV[e] = LFUDict[e]
    KV_sorted = Counter(KV)
    evict_dict = dict(KV_sorted.most_common(eviction))
    for e in C:
        if e in evict_dict:
            Y_current.append(1)
        else:
            Y_current.append(0)
    for i in range(len(Y_current)):
        if Y_current[i] is Y_OPT[i]:
            lfuCorrect+=1
        else:
            lfuIncorrect+=1
    return Y_current

# return "eviction" blocks that are being accessed furthest
# from the cache that was sent to us.

def getY(C,D):
    assert(len(C) == len(D))
    Y_current = []
    KV_sorted = Counter(D)
    evict_dict = dict(KV_sorted.most_common(eviction))
    assert(len(evict_dict) == eviction)
    all_vals = evict_dict.values()
    for e in C:
        if e in evict_dict.values():
            Y_current.append(1)
        else:
            Y_current.append(0)
    #print (Y_current.count(1))
    assert(Y_current.count(1) == eviction)
    assert((set(all_vals)).issubset(set(C)))
    return Y_current

def getLFURow(LFUDict, C):
    x_lfurow = []
    for e in C:
        x_lfurow.append(LFUDict[e])
    norm = x_lfurow / np.linalg.norm(x_lfurow)
    return norm
    
def getLRURow(LRUQ, C):
    x_lrurow = []
    KV = defaultdict(int)
    for i in range(len(LRUQ)):
        KV[LRUQ[i]] = len(C) - i
    for e in C:
        x_lrurow.append(KV[e])
    norm = x_lrurow / np.linalg.norm(x_lrurow)
    return norm

def normalize(feature, blocks):
    x_feature = []
    for i in range(len(blocks)):
        x_feature.append(feature[blocks[i]])
    return x_feature / np.linalg.norm(x_feature)

def getX(LRUQ, LFUDict, C, CacheTS, CachePID):
    X_lfurow = getLFURow(LFUDict, C)
    X_lrurow = getLRURow(LRUQ, C)
    X_bno    = C / np.linalg.norm(C)
#     X_ts     = normalize(CacheTS, C)
#     X_pid    = normalize(CachePID, C)
    return (np.column_stack((X_lfurow, X_lrurow, X_bno)))
    
# appends OPT sample to X, Y arrays

X = np.array([], dtype=np.int64).reshape(0,num_params)
Y = np.array([], dtype=np.int64).reshape(0,1)

# C - cache, LFUDict - dictionary containing block-> access frequency
# LRUQ - order of element access in Cache.

def populateData(LFUDict, LRUQ, C, D, CacheTS, CachePID):
    global X,Y
    C = list(C)
    Y_current = getY(C, D)
    X_current = getX(LRUQ, LFUDict, C, CacheTS, CachePID)

    Y = np.append(Y, Y_current)
    X = np.concatenate((X,X_current))
    assert(Y_current.count(1) == eviction)
    return Y_current

#D - dictionary for faster max() finding among available blocks
#this dictionary contains next_position -> block_number of blocks in Cache
#LFUDict - dictionary containing {block -> access_frequencies}
#LRUQ - deque of all elements in cache based on recency of access

def belady_opt(blocktrace, frame):
    global maxpos
    OPT = defaultdict(deque)
    D = defaultdict(int)
    LFUDict = defaultdict(int)
    LRUQ = []
    CacheTS = defaultdict(int)
    CachePID = defaultdict(int)

    for i, block in enumerate(tqdm(blocktrace, desc="OPT: building index")):
        OPT[block].append(i)

    hit, miss = 0, 0

    C = []
    count=0
    seq_number = 0
    for block in tqdm(blocktrace, desc="OPT"):
#    for block in blocktrace: 
        LFUDict[block] +=1

        if len(OPT[block]) is not 0 and OPT[block][0] == seq_number:
            OPT[block].popleft()
        CacheTS [blocktrace[seq_number]] = timestamp[seq_number]
        CachePID [blocktrace[seq_number]] = pid[seq_number]
        if block in C:
            hit+=1
            LRUQ.remove(block)
            LRUQ.append(block)
            assert( seq_number in D)
            del D[seq_number]
            if len(OPT[block]) is not 0:
                D[OPT[block][0]] = block
                OPT[block].popleft()
            else:
                D[maxpos] = block
                maxpos -= 1
        else:
            miss+=1
            if len(C) == frame:
                assert(len(D) == frame)
                evictpos = max(D)
                C.remove(D[evictpos])
                LRUQ.remove(D[evictpos])
                del CacheTS [D[evictpos]]
                del CachePID [D[evictpos]]
                del D[evictpos]
            if len(OPT[block]) is not 0:
                D[OPT[block][0]] = block
                OPT[block].popleft()
            else:
                D[maxpos] = block
                maxpos -= 1
            C.append(block)
            LRUQ.append(block)
            if (seq_number % sampling_freq +1 == sampling_freq and len(C) == frame):
                Y_OPT = populateData(LFUDict, LRUQ, C, D, CacheTS, CachePID)
                lruPredict(C,LRUQ,Y_OPT)
                lfuPredict(C,LFUDict,Y_OPT)
        seq_number += 1

    hitrate = hit / (hit + miss)
    print(hitrate)
    return hitrate

belady_opt(blocktrace, cache_size)

print ("size of X " + str(len(X)))

# round off so that train, test splits are cache size aligned
X = X[0:len(X)-(len(X)%(cache_size * 10))]
Y = Y[0:len(Y)-(len(Y)%(cache_size * 10))]

print ("Test Y")

for i in range(int(len(X) / 1000)):
   y = Y[i*1000:(i+1) *1000]
   assert(Counter(y)[1] == eviction)

print ("size of X " + str(len(X)))
print ("size of Y " + str(len(Y)))

#Train-Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size=0.3, random_state=0, shuffle=False)

print ("Test Y_test")

for i in range(int(len(X_test) / cache_size)):
   y = Y_test[i*cache_size:(i+1) *cache_size]
   assert(Counter(y)[1] == eviction)

print ("size of X_train " + str(len(X_train)))
print ("size of X_test " + str(len(X_test)))

#Fitting Logistic Regression Model
#logreg = LogisticRegression(solver='lbfgs')
#‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
logreg = LogisticRegression(solver='saga')
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
print("======================================")
print(logreg.predict_proba([X_test[0]]))
print(Y_test[0])
print("=======================================")
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

#confusion_matrix = confusion_matrix(Y_test,Y_pred)
#print (confusion_matrix)

print (logreg.coef_)

print ("LFU Correct / Incorrect Ratio")
total = lfuCorrect + lfuIncorrect
print ( lfuCorrect / total )

print ("LRU Correct / Incorrect Ratio")
total = lruCorrect + lruIncorrect
print ( lruCorrect / total )

c=0

logRegIncorrect = 0
logRegCorrect = 0

for i in range(int(len(X_test)/cache_size)):
    Y_pred_prob = logreg.predict_proba(X_test[i*cache_size:(i+1)*cache_size])
    Y_pred_current = Y_getMinPredict(Y_pred_prob)
    Y_test_current = Y_test[i*cache_size:(i+1)*cache_size]
    assert(Counter(Y_test_current)[1] == eviction)
    for j in range(len(Y_test_current)):
        if np.equal(Y_test_current[j], Y_pred_current[j]):
            logRegCorrect +=1
        else:
            logRegIncorrect +=1

print ("logRegCorrect = " + str(logRegCorrect))
print ("logRegInorrect = " + str(logRegIncorrect))
print ("correct = " + str(logRegCorrect / ( logRegCorrect + logRegIncorrect)))


def hitRate(blocktrace, frame):
    LFUDict = defaultdict(int)
    LRUQ = []
    CacheTS = defaultdict(int)
    CachePID = defaultdict(int)

    hit, miss = 0, 0

    C = []
    evictCacheIndex = np.array([])
    count=0
    seq_number = 0
    for block in tqdm(blocktrace, desc="OPT"):
        LFUDict[block] +=1
        CacheTS [blocktrace[seq_number]] = timestamp[seq_number]
        CachePID [blocktrace[seq_number]] = pid[seq_number]
        if block in C:
            hit+=1
            #if C.index(block) in evictCacheIndex:
            #    np.delete(evictCacheIndex, C.index(block))
                
            LRUQ.remove(block)
            LRUQ.append(block)
        else:
            evictPos = -1
            miss+=1
            if len(C) == frame:
                if len(evictCacheIndex) == 0: # call eviction candidates
                    X_test = getX(LRUQ, LFUDict, C, CacheTS, CachePID)
                    Y_pred_prob = logreg.predict_proba(X_test)
                    # index of cache blocks that should be removed
                    evictCacheIndex = Y_getBlockSeq(Y_pred_prob)

                # evict from cache
                evictPos = evictCacheIndex[0]
                evictBlock = C[evictPos]
                LRUQ.remove(evictBlock)
                del CacheTS [evictBlock]
                del CachePID [evictBlock]
            if evictPos is -1:
                C.append(block)
            else:
                C[evictPos] = block
                np.delete(evictCacheIndex, 0)
            LRUQ.append(block)
            CacheTS [blocktrace[seq_number]] = timestamp[seq_number]
            CachePID [blocktrace[seq_number]] = pid[seq_number]
        seq_number += 1

    hitrate = hit / (hit + miss)
    print(hitrate)
    return hitrate

x = blocktrace[-int(0.3 * len(blocktrace)):]

belady_opt(x, cache_size)
hitRate(x, cache_size)
LFU(x, cache_size)
# get LFU hit rate.!!!!!
# OPT HIT RATE: 0.07700060725633524


