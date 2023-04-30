import itertools
import os

models = ['TGN']
datasets = ['GDELT']
strategies = ['naive',  'fix', 'lineardeg_adaptive']

for model, dataset, strategy in itertools.product(models, datasets, strategies):
    os.system(
        f'python benchmark_sampler.py --model {model} --dataset {dataset} --adaptive-block-size-strategy {strategy}")
