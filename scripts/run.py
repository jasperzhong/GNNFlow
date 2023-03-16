import itertools
import os

models = ['TGN']
cache_ratio = [0.2, 0.4]
delay_epoch = [0, 2]
cache_min_neighbor = [10]

param_space = itertools.product(
    models, cache_ratio, delay_epoch, cache_min_neighbor)

# for param in param_space:
#     print(param)
#     os.system("python3 offline_edge_prediction.py --model {} --data REDDIT --cache LRUCache --ingestion-batch-size 10000000  --node-embed-cache-ratio {} --cache-delay-epoch {} --cache-min-neighbor {}".format(*param))

for model in models:
    os.system("python3 offline_edge_prediction.py --model {} --data REDDIT --cache LRUCache --ingestion-batch-size 10000000  --node-embed-cache-ratio 0".format(model))

"""
1. cache hit rate
2. val ap/auc 
3. throughput 
"""
