import itertools
import os

datasets = ['NETFLIX']
# strategies = ['naive', 'linearavg', 'lineardeg', 'lineardeg_adaptive']
# strategies = ['naive',  'fix', 'lineardeg_adaptive']
strategies = ['fix']


bigblock_sizes = {
    "WIKI": 64,
    "REDDIT": 128,
    "MOOC": 32,
    "LASTFM": 512,
    "GDELT": 1024,
    "MAG": 8196,
}

block_sizes = [32]

for dataset, strategy, bigblock_size in itertools.product(datasets, strategies, block_sizes):
    os.system(
        f'BIGBLOCK_SIZE={bigblock_size} python benchmark_build_graph.py --dataset {dataset} --adaptive-block-size-strategy {strategy} --ingestion-batch-size 100000')
