from logging import log
from operator import ge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline, make_interp_spline

plt.rcParams.update({'font.size': 26, 'font.family': 'Myriad Pro'})

plt.figure(figsize=(12, 9))

dataset = "wiki"

def load_data(file):
    retrain = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            retrain.append(float(line))

    retrain = np.array(retrain) 

    return retrain

plt.title(dataset)
plt.ylabel('test AP')
# plt.ylim(0.48, 0.72)
# plt.ylim(0.95, 1.0)
plt.ylim(0.88, 0.98)
plt.xlabel('adding #edges (k) ')

no_retrain = load_data("{}_no.txt".format(dataset))
retrain21 = load_data("{}_21.txt".format(dataset))
retrain51 = load_data("{}_51.txt".format(dataset))
iter_500 = 500

linewidth = 5

plt.plot(np.arange(len(no_retrain)), no_retrain, linewidth=linewidth, label="no retrain")
plt.plot(np.arange(len(retrain21)), retrain21, linewidth=linewidth, label="retrain every 20k")
plt.plot(np.arange(len(retrain51)), retrain51, linewidth=linewidth, label="retrain every 50k")
# plt.xlim(0.85, 1.0)

plt.legend()

# plt.savefig("e2e_bert.svg", dpi=400, bbox_inches='tight')
plt.savefig("{}.png".format(dataset), dpi=400, bbox_inches='tight')