import itertools
import os
import torch
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading

def run_command(command):    
    process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(command, process.stdout.decode().strip())

def run():
    param_space = itertools.product(
        models, datasets, caches)
    commands = []
    for i, param in enumerate(param_space):
        device = i % torch.cuda.device_count()
        commands.append('CUDA_VISIBLE_DEVICES={} python3 benchmark_cache.py --model {} --dataset {} --cache {} --node-cache-ratio 0.01 --edge-cache-ratio 0.001'.format(device, *param))
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        results = list(executor.map(run_command, commands))


models = ['TGN', 'TGAT', 'GRAPHSAGE']
datasets = ['WIKI', 'REDDIT', 'MOOC', 'LASTFM']
caches = ['LRUCache', 'LFUCacue', 'LFUDACache', 'GNNLabStaticCache', 'FIFOCache']

if __name__ == '__main__':
    run()