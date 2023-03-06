import itertools
import os

cache_ratio = [0.2, 0.4]
max_staleness = [1800, 3600]
delay_epoch = [0, 1, 2]
delay_iter = [0, 50, 100]

param_space = itertools.product(cache_ratio, max_staleness, delay_epoch, delay_iter)


for param in param_space:
    print(param)
    os.system("python3 offline_edge_prediction.py --model TGAT --data REDDIT --cache LRUCache --ingestion-batch-size 10000000  --node-embed-cache-ratio {} --max-staleness {} --cache-delay-epoch {} --cache-delay-iter {}".format(*param))

os.system("python3 offline_edge_prediction.py --model TGAT --data REDDIT --cache LRUCache --ingestion-batch-size 10000000  --node-embed-cache-ratio 0")
"""
1. cache hit rate
2. val ap/auc 
3. throughput 
"""
