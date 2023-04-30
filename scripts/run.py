import itertools
import os

models = ['TGN']
datasets = ['GDELT']
n_gpus = [8]

for model, dataset, n_gpu in itertools.product(models, datasets, n_gpus):
    print(f'Running {model} on {dataset} with {n_gpu} GPUs')
    os.system(
        f'./run_offline.sh {model} {dataset} LRUCache 0.03 1 {n_gpu}')
