import itertools
import os

models = ['TGN', 'TGAT', 'GRAPHSAGE']
datasets = ['MOOC', 'LASTFM']
cache = ['LRUCache', 'LFUCache', 'FIFOCache', 'GNNLabStaticCache']
param_space = itertools.product(
    models, datasets, cache)

for param in param_space:
    os.system("python3 offline_edge_prediction.py --model {} --data {} --epoch 10 --cache {} --edge-cache-ratio 0.2 --node-cache-ratio 0".format(*param))

param_space = itertools.product(
    models, datasets)
