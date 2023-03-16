import itertools
import os

models = ['TGN', 'TGAT', 'GRAPHSAGE']
datasets = ['WIKI', 'MOOC', 'LASTFM']

param_space = itertools.product(
    models, datasets)

for param in param_space:
    print(param)
    os.system(
        "python3 sampling_freq.py --model {} --dataset {} --stat".format(*param))
