#!/usr/bin/env python
# coding: utf-8

# In[458]:


from tqdm import tqdm as tqdm 
import numpy as np
from collections import deque, defaultdict
import timeit
import pandas as pd
import heapq
import heapq_max

# In[459]:


df = pd.read_csv('cheetah.cs.fiu.edu-110108-113008.1.blkparse', sep=' ',header = None)
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

H = []
#heapq_max.heapify(H) 

def getFurthestAccessBlock(C, OPT):
    maxAccessPosition = -1
    maxAccessBlock = -1
    for cached_block in C:
        if len(OPT[cached_block]) is 0:
            return cached_block            
    for cached_block in C:
        if OPT[cached_block][0] > maxAccessPosition:
            maxAccessPosition = OPT[cached_block][0]
            maxAccessBlock = cached_block
    return maxAccessBlock

def getEvictedBlocks(C, percentage):
    return 0

def Populate(C, C_recency, C_frequency, seqNo):
    Y = getEvictedBlocks(C, percentage)
    # X gets populated here.
    return 0

def belady_opt(blocktrace, frame):
    global H
    OPT = defaultdict(deque)

    for i, block in enumerate(tqdm(blocktrace, desc="OPT: building index")):
        OPT[block].append(i)    

    #print ("created OPT dictionary")    

    hit, miss = 0, 0

    blockCount = defaultdict(int)
    C = set()
    seq_number = 0
    for block in tqdm(blocktrace, desc="OPT"):
        blockCount[block] +=1
        if block in C:
            #OPT[block] = OPT[block][1:]
            hit+=1
            #print('hit' + str(block))
            #print(OPT)
            OPT[block].popleft()
        else:
            #print('miss' + str(block))
            miss+=1
            if len(C) == frame:
                fblock = getFurthestAccessBlock(C, OPT)
                assert(fblock != -1)
                C.remove(fblock)
            C.add(block)
            #OPT[block] = OPT[block][1:]
            #print(OPT)
            OPT[block].popleft()

    #print ("hit count" + str(hit_count))
    #print ("miss count" + str(miss_count))
    hitrate = hit / (hit + miss)
    print(hitrate)
    return hitrate


# In[455]:


belady_opt(blocktrace, 500)


# In[ ]:

