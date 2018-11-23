#!/usr/bin/env python
# coding: utf-8

# In[458]:

from tqdm import tqdm as tqdm 
import numpy as np
from collections import deque, defaultdict
import timeit
import pandas as pd
import random

max_block = -100000000000000

# In[459]:


df = pd.read_csv('cheetah.cs.fiu.edu-110108-113008.1.blkparse', sep=' ',header = None)
#df = pd.read_csv('sample.blkparse', sep=' ',header = None)
df.columns = ['timestamp','pid','pname','blockNo', 'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']
df.head()


# In[460]:


blocktrace = df['blockNo'].tolist()
len(blocktrace)


# In[461]:


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


# In[468]:


#LFU(blocktrace, 500)


# In[454]:

def belady_opt(blocktrace, frame):
    OPT = defaultdict(deque)
    D = defaultdict(int)

    for i, block in enumerate(tqdm(blocktrace, desc="OPT: building index")):
        OPT[block].append(i)

    hit, miss = 0, 0

    blockCount = defaultdict(int)
    C = set()
    count=0
    seq_number = 0
    for block in tqdm(blocktrace, desc="OPT"):
        blockCount[block] +=1

        if len(OPT[block]) is not 0 and OPT[block][0] == seq_number:
            OPT[block].popleft()
        if block in C:
            hit+=1
            if seq_number in D:
                del D[seq_number]
                if len(OPT[block]) is not 0:
                    D[OPT[block][0]] = block
                    OPT[block].popleft()
        else:
            miss+=1
            if len(C) == frame:
                if(len(D) != 0):
                    evictpos = max(D)
                    C.remove(D[evictpos])
                    del D[evictpos]
                else:
                    evictpos = 0
                    C.remove(random.sample(C,1)[0])
            if len(OPT[block]) is not 0:
                D[OPT[block][0]] = block
                OPT[block].popleft()
            C.add(block)
        seq_number += 1

    hitrate = hit / (hit + miss)
    print(hitrate)
    return hitrate

# In[455]:

belady_opt(blocktrace, 500)


# In[ ]:
