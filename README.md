# Applying Machine Learning on Cache Replacement Policy (commonly known as Caching Algorithm)

Course Project for [CSC2514/CSC411](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/ "Course Website")

### Abstract

Enterprise data centers store persistent data in different storage media, varying in speed and capacity. Reducing overall latency incurred in serving requests involves migrating frequently accessed hot data from slower to faster media (fetching) and moving colder data from faster to slower media (eviction). Managing data involves conventional, hard-coded algorithms that either do not reflect the access patterns of enterprise workloads (LRU, LFU, FIFO) or are practically unimplementable (OPT). In this paper, we analyze real-world workload traces and evaluate 3 machine learning based approaches - Logistic Regression, Neural Network and k-NN for block eviction over hybrid media. Second, we propose two novel hyper parameters - sampling_frequency and evict count for cache replacement policy. Third, we architect two data models - Block Cache and Vectorized Cache for block classification. Finally, our proposed block eviction technique is scalable across cache sizes when a parametric Machine Learning algorithm is used.
