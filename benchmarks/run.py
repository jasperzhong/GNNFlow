import itertools
import os

models = ['GRAPHSAGE']
datasets = ['REDDIT', 'MOOC', 'LASTFM']

for model, dataset in itertools.product(models, datasets):
    os.system(
        f'python benchmark_sampler.py --model {model} --dataset {dataset}')
