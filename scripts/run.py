import itertools
import os

models = ['TGN', 'TGAT', 'DySAT']
datasets = ['REDDIT', 'LASTFM']
mem_resource_types = ['cuda', 'pinned', 'unified', 'shared']

for model, dataset, mem_resource_type in itertools.product(models, datasets, mem_resource_types):
    print(f'Running {model} on {dataset} with {mem_resource_type} GPUs')
    os.system(
        f'./run_offline.sh {model} {dataset} LRUCache 0.03 1 1 {mem_resource_type}')
