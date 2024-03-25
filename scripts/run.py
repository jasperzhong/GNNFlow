import itertools
import os

import numpy as np

models = ['TGN']
datasets = ['REDDIT']
caches = ['LRUCache']  # 'GNNLabStaticCache']
# caches = ['GNNLabStaticCache']
reset_caches = [1]
edge_cache_ratios = [0.01, 0.02, 0.03, 0.04, 0.05]

for model, dataset, cache, reset_cache, edge_cache_ratio in itertools.product(models, datasets, caches, reset_caches, edge_cache_ratios):
    # print(
    #     f'Running {model} on {dataset} with {cache} and reset_cache={reset_cache} and edge_cache_ratio={edge_cache_ratio}')
    # os.system('rm -rf /dev/shm/rmm_*')
    # cmd = f"python online_edge_prediction.py --model {model} --data {dataset} --cache {cache} --edge-cache-ratio {edge_cache_ratio/10} --node-cache-ratio {edge_cache_ratio} --replay-ratio 0 --phase1-ratio 0.3 --snapshot-time-window 0 --epoch 1"
    # if reset_cache:
    #     cmd += ' --reset-cache'
    # print(cmd)
    # os.system(cmd)

    reset_cache = bool(reset_cache)
    prefix = f'{model}_{dataset}_0.0_1_{cache}_{edge_cache_ratio/10:.5f}_{edge_cache_ratio:.5f}_reset{reset_cache}'
    rate = np.mean(
        np.load("tmp_res/continuous/{}_edge_cache_hit_rate.npy".format(prefix)))
    print(rate)
