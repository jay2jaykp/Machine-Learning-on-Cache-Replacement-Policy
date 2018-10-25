#!/usr/bin/python3

'''
	OPT / Belady's algorithm for cache eviction.

	This algorithm simulates the OPT cache eviction policy.
	Input: block trace file containing a column with multiple different Blocks.
	Output: Cache Miss count and cache configuration.
	
	Description
	
	We first generate a dictionary of block numbers and their relative access sequence.
	We call this dictionary the OPT dictionary.

	[Block Number1] [Position1, Postion2 ...]
	[Block Number2] [Position1, Postion2 ...]
	[Block Number3] [Position1, Postion2 ...]

	We create an array of Cached blocks "C" which is empty in the beginning.

	for each block request:
		if the block is present in the C, remove the position from the OPT dictionary.
		hit_count++
		if the block is not present in C, find max_distance
			max_distance = max(OPT[Bnumber1][0], OPT[BNumber2][0], OPT[BNumber3][0])
			where {Bnumber1, Bnumber2...} E C
		remove max_distance block from C, remove position of requested block from OPT dictionary.
		miss_count++

	hit_count+miss_count = sizeof block_request list
		
'''

import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm

import argparse

block_trace = []

'''
get the furthest accessed block. Scans OPT dictionary and selects the maximum positioned element
'''

def getFurthestAccessBlock():
    global OPT
    global C
    maxAccessPosition = -1
    maxAccessBlock = -1
    for cached_block in C:
        if len(OPT[cached_block]) is 0:
            #print ( "Not Acccessing block anymore " + str(cached_block))
            return cached_block            
        if OPT[cached_block][0] > maxAccessPosition:
            maxAccessPosition = OPT[cached_block][0]
            maxAccessBlock = cached_block
    #print ( "chose to evict " + str(maxAccessBlock) + " since its position of access is " + str(maxAccessPosition))
    return maxAccessBlock

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OPT.\n')
    parser.add_argument('cache_size', type=int,  help='Size of the Cache\n')
    parser.add_argument('idx', type=int,  help='column number of blocks. type 0 if only 1 column containing block trace is present\n')
    parser.add_argument('traceFile', type=str,  help='trace file name\n')
    args = parser.parse_args() 

    cache_size = args.cache_size
    idx = args.idx
    traceFile = args.traceFile

    with open(traceFile) as f:
        for line in f:
            block_trace.append(int(line.split()[idx]))

    print ("created block trace list")
    blockTraceLength = len(block_trace)
    # build OPT 
    
    OPT = defaultdict(partial(np.ndarray,0))
    
    seq_number = 0
    for b in block_trace:
        OPT[b] = np.append(OPT[b],seq_number)
        seq_number+=1
    
    print ("created OPT dictionary")    
#    print (OPT)
    # run algorithm
    
    hit_count = 0
    miss_count = 0
    
    C = set()
    
    seq_number = 0
    for b in tqdm(block_trace):
        seq_number+=1
        if(seq_number % (blockTraceLength / 10) == 0):
            print("Completed "+str(( seq_number * 100 / blockTraceLength)) + " %")
        if b in C:
            #print ("HIT " + str(b))
#            np.delete(OPT[b],[0])
          #  OPT[b] = OPT[b][1:]
            OPT[b] = np.delete(OPT[b],0)
            hit_count+=1
        else:
        #    print ("MISS " + str(b))
            miss_count+=1
            if len(C) == cache_size:
                fblock = getFurthestAccessBlock()
                assert(fblock != -1)
                C.remove(fblock)
            C.add(b)
      #      np.delete(OPT[b],[0])
      #      OPT[b] = OPT[b][1:]
            OPT[b] = np.delete(OPT[b],0)
       #     print ("CACHE ")
       #     print (C)
    
    print ("hit count" + str(hit_count))
    print ("miss count" + str(miss_count))
    
    #print ("Cache")
    #print (C)
