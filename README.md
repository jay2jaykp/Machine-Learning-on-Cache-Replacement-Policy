# Applying Machine Learning on Cache Replacement Policy (commonly known as Caching Algorithm)

Course Project for [CSC2514/CSC411](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/ "Course Website")

### Abstract

Enterprise data centers store persistent data in different storage media, varying in speed and capacity. Reducing overall latency incurred in serving requests involves migrating frequently accessed hot data from slower to faster media (fetching) and moving colder data from faster to slower media (eviction). Managing data involves conventional, hard-coded algorithms that either do not reflect the access patterns of enterprise workloads (LRU, LFU, FIFO) or are practically unimplementable (OPT). In this paper, we analyze real-world workload traces and evaluate 3 machine learning based approaches - Logistic Regression, Neural Network and k-NN for block eviction over hybrid media. Second, we propose two novel hyper parameters - sampling_frequency and evict count for cache replacement policy. Third, we architect two data models - Block Cache and Vectorized Cache for block classification. Finally, our proposed block eviction technique is scalable across cache sizes when a parametric Machine Learning algorithm is used.

### Important Notebooks

This repo is used very roughly throughout the course, please see the following notebooks which are curated for public.

[1] [Standard Caching Algorithm (written in Python 3)](https://github.com/jay2jaykp/Machine-Learning-on-Cache-Replacement-Policy/blob/master/Standard%20Caching%20Algorithms.ipynb)

[2] [Project Code](https://github.com/jay2jaykp/Machine-Learning-on-Cache-Replacement-Policy/blob/master/Towards%20ML%20based%20Cache%20Management.ipynb)

[3] [Project Report](https://www.overleaf.com/read/jsjcnwrvxdpr)

### Contributors:

#### [1] Shehbaz Jaffer <http://www.cs.toronto.edu/~shehbaz/>
#### [2] Maharshi Trivedi <https://nonlinear.mie.utoronto.ca/maharshi/>
#### [3] Jay Patel
