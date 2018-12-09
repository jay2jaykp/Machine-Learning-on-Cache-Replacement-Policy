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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import decomposition

# dummy maxmimum position variable. assign the position of blocks that 
# will never get accessed a value greater than this value. this way OPT
# can be fooled to think that the block will be accessed but at a position
# far-far-away in time.

maxpos = 1000000000000

num_params = 5
sampling_freq = 1000 # number of samples skipped
cache_size = 1000    # default cache size
eviction = int(0.1 * cache_size)  # number of blocks evicted
filename = "cheetah.cs.fiu.edu-110108-113008.1.blkparse"
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
    X_bno    = C / np.linalg.norm(C)
    X_ts     = normalize(CacheTS, C)
    X_pid    = normalize(CachePID, C)
    X_lrurow = getLRURow(LRUQ, C)
    return (np.column_stack((X_lfurow, X_bno, X_ts, X_pid, X_lrurow)))
    
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

    C = set()
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
            C.add(block)
            LRUQ.append(block)
            if (seq_number % sampling_freq +1 == sampling_freq and len(C) == frame):
                Y_OPT = populateData(LFUDict, LRUQ, C, D, CacheTS, CachePID)
        seq_number += 1

    hitrate = hit / (hit + miss)
    print(hitrate)
    return hitrate

belady_opt(blocktrace, cache_size)

pca = decomposition.PCA(n_components=2)

pc = pca.fit_transform(X)
#i = np.identity(df.shape[1])
#coef = pca.transform(i)

print (pca.components_)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

pc_df = pd.DataFrame(data = pc , 
                columns = ['PC1', 'PC2'])
pc_df['Cluster'] = Y

plot = sns.lmplot( x="PC1", y="PC2", data=pc_df, fit_reg=False, 
    hue='Cluster', # color by cluster
      legend=True,
        scatter_kws={"s": 80}) # specify the point size

plt.show()
plt.savefig('pca.png')

#pc_df = pd.DataFrame(data = pc , 
#                columns = ['PC1', 'PC2','PC3','PC4'])
#pc_df['Cluster'] = Y
#
#df = pd.DataFrame({'var':pca.explained_variance_ratio_,
#                 'PC':['LFU','LRU','BNO','TimeStamp']})
#plot = sns.barplot(x='PC',y="var", data=df, color="c")
#plt.show()
#plt.savefig('pca.png')
#
#print ("size of X " + str(len(X)))
#
## round off so that train, test splits are cache size aligned
#X = X[0:len(X)-(len(X)%(cache_size * 10))]
#Y = Y[0:len(Y)-(len(Y)%(cache_size * 10))]
#
#print ("Test Y")
#
#for i in range(int(len(X) / 1000)):
#    y = Y[i*1000:(i+1) *1000]
#    assert(Counter(y)[1] == eviction)
#
#print ("size of X " + str(len(X)))
#print ("size of Y " + str(len(Y)))
#
##Train-Test split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size=0.3, random_state=0, shuffle=False)
#
#print ("Test Y_test")
#
#for i in range(int(len(X_test) / cache_size)):
#    y = Y_test[i*cache_size:(i+1) *cache_size]
#    assert(Counter(y)[1] == eviction)
#
#print ("size of X_train " + str(len(X_train)))
#print ("size of X_test " + str(len(X_test)))
#
##Fitting Logistic Regression Model
##logreg = LogisticRegression(solver='lbfgs')
##‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
#logreg = LogisticRegression(solver='saga')
#logreg.fit(X_train, Y_train)
#
#Y_pred = logreg.predict(X_test)
#
##print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))
#
##confusion_matrix = confusion_matrix(Y_test,Y_pred)
##print (confusion_matrix)
#
#print (logreg.coef_)
#
#print ("LFU Correct / Incorrect Ratio")
#total = lfuCorrect + lfuIncorrect
#print ( lfuCorrect / total )
#
#print ("LRU Correct / Incorrect Ratio")
#total = lruCorrect + lruIncorrect
#print ( lruCorrect / total )
#
#c=0
#
#logRegIncorrect = 0
#logRegCorrect = 0
#
#for i in range(int(len(X_test)/cache_size)):
#    Y_pred_prob = logreg.predict_proba(X_test[i*cache_size:(i+1)*cache_size])
#    Y_pred_current = Y_getMinPredict(Y_pred_prob)
#    Y_test_current = Y_test[i*cache_size:(i+1)*cache_size]
#    assert(Counter(Y_test_current)[1] == eviction)
#    for j in range(len(Y_test_current)):
#        if np.equal(Y_test_current[j], Y_pred_current[j]):
#            logRegCorrect +=1
#        else:
#            logRegIncorrect +=1
#
#print ("logRegCorrect = " + str(logRegCorrect))
#print ("logRegInorrect = " + str(logRegIncorrect))
#print ("correct = " + str(logRegCorrect / ( logRegCorrect + logRegIncorrect))) 
