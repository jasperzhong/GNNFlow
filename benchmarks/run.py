import itertools
import os

models = ['shuffle_TGAT']
datasets = ['WIKI', 'REDDIT', 'MOOC', 'LASTFM']

param_space = itertools.product(
    models, datasets)

for param in param_space:
    print(param)
    os.system(
        "python3 sampling_freq2.py --model {} --dataset {} --stat".format(*param))
