import itertools
import os

datasets = ['WIKI', 'REDDIT', 'MOOC', 'LASTFM']
strategies = ['naive', 'linearadt', 'linearroundadt', 'lineardeg', 'logdeg']

for dataset, strategy in itertools.product(datasets, strategies):
    os.system(
        f'python benchmark_build_graph.py --dataset {dataset} --adaptive-block-size-strategy {strategy}')
