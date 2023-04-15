import itertools
import os

datasets = ['WIKI', 'REDDIT', 'MOOC', 'LASTFM']
strategies = ['naive', 'linearavg', 'lineardeg', 'lineardeg_adaptive']

bigblock_thresholds = {
    "WIKI": 32,
    "REDDIT": 64,
    "MOOC": 16,
    "LASTFM": 64,
}

bigblock_sizes = {
    "WIKI": 64,
    "REDDIT": 128,
    "MOOC": 32,
    "LASTFM": 256,
}

for dataset, strategy in itertools.product(datasets, strategies):
    bigblock_threshold = bigblock_thresholds[dataset]
    bigblock_size = bigblock_sizes[dataset]
    os.system(
        f'BIGBLOCK_THRESHOLD={bigblock_threshold} BIGBLOCK_SIZE={bigblock_size} python benchmark_build_graph.py --dataset {dataset} --adaptive-block-size-strategy {strategy}')
