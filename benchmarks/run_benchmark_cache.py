import itertools
import os
import numpy as np

models = ['shuffle_GRAPHSAGE', 'GRAPHSAGE']
datasets = ['WIKI', 'REDDIT', 'MOOC', 'LASTFM']
cache_ratio = np.arange(0, 1.1, 0.1)

param_space = itertools.product(
    models, datasets, cache_ratio)

for param in param_space:
    print(param)
    os.system(
        "python3 benchmark_cache.py --model {} --dataset {} --cache LRUCache --node-cache-ratio {:.1f} --edge-cache-ratio {:.1f} --shuffle".format(*param, param[-1]))
