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
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# dummy maxmimum position variable. assign the position of blocks that 
# will never get accessed a value greater than this value. this way OPT
# can be fooled to think that the block will be accessed but at a position
# far-far-away in time.

maxpos = 1000000000000

num_params = 3
sampling_freq = 1000 # number of samples skipped
eviction = 100       # number of blocks evicted
cache_size = 500    # default cache size
filename = "cheetah.cs.fiu.edu-110108-113008.1.blkparse"
#filename = "sample.blkparse"

df = pd.read_csv(filename, sep=' ',header = None)
df.columns = ['timestamp','pid','pname','blockNo', 'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']
df.head()

# In[460]:

blocktrace = df['blockNo'].tolist()
len(blocktrace)

def FIFO(blocktrace, frame):
    
    cache = deque(maxlen=frame)
    hit, miss = 0, 0
    
    for block in tqdm(blocktrace, leave=False):
        
        if block in cache:
            hit += 1

        else:
            cache.append(block)
            miss += 1
    
    hitrate = hit / (hit+miss)
    return hitrate 


# In[462]:


#FIFO(blocktrace, 50)


# In[463]:


def LIFO(blocktrace, frame):
    
    cache = deque(maxlen=frame)
    hit, miss = 0, 0
    
    for block in tqdm(blocktrace, leave=False):
        if block in cache:
            hit += 1
            
        elif len(cache) < frame:
            cache.append(block)
            miss += 1
        
        else:
            cache.pop()
            cache.append(block)
            miss += 1
            
    hitrate = hit / (hit + miss)
    return hitrate


# In[464]:


#LIFO(blocktrace, 50)


# In[465]:


def LRU(blocktrace, frame):
    
    cache = set()
    recency = deque()
    hit, miss = 0, 0
    
    for block in tqdm(blocktrace, leave=False):
        
        if block in cache:
            recency.remove(block)
            recency.append(block)
            hit += 1
            
        elif len(cache) < frame:
            cache.add(block)
            recency.append(block)
            miss += 1
            
        else:
            cache.remove(recency[0])
            recency.popleft()
            cache.add(block)
            recency.append(block)
            miss += 1
    
    hitrate = hit / (hit + miss)
    return hitrate


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

def getY(C,OPT):
    global maxpos
    KV = defaultdict(int)
    Y_current = []
    for e in C:
        if len(OPT[e]) is not 0:
            KV[e] = OPT[e][0]
        else:
            KV[e] = maxpos
            maxpos+=1
    # extract "eviction" blocks from KV_sorted hashmap
    KV_sorted = Counter(KV)
    evict_dict = dict(KV_sorted.most_common(eviction))
    for e in C:
        if e in evict_dict:
            Y_current.append(1)
        else:
            Y_current.append(0)
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

def getX(LRUQ, LFUDict, C):
    X_lfurow = getLFURow(LFUDict, C)
    X_lrurow = getLRURow(LRUQ, C)
    X_bno    = C / np.linalg.norm(C)
    return (np.column_stack((X_lfurow, X_lrurow, X_bno)))

# appends OPT sample to X, Y arrays

X = np.array([], dtype=np.int64).reshape(0,num_params)
Y = np.array([], dtype=np.int64).reshape(0,1)

# C - cache, LFUDict - dictionary containing block-> access frequency
# LRUQ - order of element access in Cache.

def populateData(LFUDict, LRUQ, C, OPT):
    global X,Y
    C = list(C)
    Y_current = getY(C, OPT)
    X_current = getX(LRUQ, LFUDict, C)

    Y = np.append(Y, Y_current)
    X = np.concatenate((X,X_current))
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

    for i, block in enumerate(tqdm(blocktrace, desc="OPT: building index")):
        OPT[block].append(i)

    hit, miss = 0, 0

    C = set()
    count=0
    seq_number = 0
    for block in tqdm(blocktrace, desc="OPT"):
        LFUDict[block] +=1

        if len(OPT[block]) is not 0 and OPT[block][0] == seq_number:
            OPT[block].popleft()
        if block in C:
            hit+=1
            LRUQ.remove(block)
            LRUQ.append(block)
            if seq_number in D:
                del D[seq_number]
                if len(OPT[block]) is not 0:
                    D[OPT[block][0]] = block
                    OPT[block].popleft()
                else:
                    D[maxpos] = block
                    maxpos+=1
        else:
            miss+=1
            if len(C) == frame:
                if(len(D) != 0):
                    evictpos = max(D)
                    C.remove(D[evictpos])
                    LRUQ.remove(D[evictpos])
                    del D[evictpos]
            if len(OPT[block]) is not 0:
                D[OPT[block][0]] = block
                OPT[block].popleft()
            else:
                D[maxpos] = block
                maxpos+=1
            C.add(block)
            LRUQ.append(block)
            if (seq_number % sampling_freq +1 == sampling_freq):
                Y_OPT = populateData(LFUDict, LRUQ, C, OPT)
                lruPredict(C,LRUQ,Y_OPT)
                lfuPredict(C,LFUDict,Y_OPT)
        seq_number += 1

    hitrate = hit / (hit + miss)
    print(hitrate)
    return hitrate

belady_opt(blocktrace, cache_size)

#Train-Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size=0.3, random_state=0)

#Fitting Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

confusion_matrix = confusion_matrix(Y_test,Y_pred)
print (confusion_matrix)

print ("LFU Correct / Incorrect Ratio")
total = lfuCorrect + lfuIncorrect
print ( (total - (lfuIncorrect/2) ) / total )

print ("LRU Correct / Incorrect Ratio")
total = lruCorrect + lruIncorrect
print ( (total - (lruIncorrect/2) ) / total )
