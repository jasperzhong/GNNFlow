import itertools
import os

models = ['TGN', 'TGAT', 'DySAT']
datasets = ['REDDIT', 'LASTFM']
mem_resource_types = ['cuda', 'pinned', 'unified', 'shared']

for dataset, mem_resource_type in itertools.product(datasets, mem_resource_types):
    cmd = f'python benchmark_sampler.py --model {" ".join(models)} --dataset {dataset} --mem-resource-type {mem_resource_type} --ingestion-batch-size 100000'
    print(cmd)
    os.system(cmd)
        
