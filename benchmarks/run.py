import itertools
import os

models = ['TGN', 'TGAT', 'DySAT', 'GRAPHSAGE']
datasets = ['WIKI', 'REDDIT', 'MOOC', 'LASTFM']

for model, dataset in itertools.product(models, datasets):
    os.system(
        f'python benchmark_sampler.py --model {model} --dataset {dataset}')
    os.system(
        f'python benchmark_sampler.py --model {model} --dataset {dataset} --sort')
