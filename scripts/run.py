import itertools
import os

models = ['TGN']
datasets = ['REDDIT']
n_gpus = [1, 2, 4, 8]

for model, dataset, n_gpu in itertools.product(models, datasets, n_gpus):
    print(f'Running {model} on {dataset} with {n_gpu} GPUs')
    os.system(
        f'./run_offline.sh {model} {dataset} LRUCache 0.03 1 0 {n_gpu}')
