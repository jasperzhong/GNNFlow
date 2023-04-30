import itertools
import os

datasets = ['LASTFM']
# strategies = ['naive', 'linearavg', 'lineardeg', 'lineardeg_adaptive']
# strategies = ['naive',  'fix', 'lineardeg_adaptive']
strategies = ['lineardeg_adaptive']


bigblock_sizes = {
    "WIKI": 64,
    "REDDIT": 128,
    "MOOC": 32,
    "LASTFM": 512,
    "GDELT": 1024,
    "MAG": 8196,
}

for dataset, strategy in itertools.product(datasets, strategies):
    bigblock_size = bigblock_sizes[dataset]
    os.system(
        f'BIGBLOCK_SIZE={bigblock_size} python benchmark_build_graph.py --dataset {dataset} --adaptive-block-size-strategy {strategy} --ingestion-batch-size 10000')
