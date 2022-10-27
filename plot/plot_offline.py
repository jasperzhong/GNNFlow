from logging import log
from operator import ge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline, make_interp_spline

plt.rcParams.update({'font.size': 26, 'font.family': 'Myriad Pro'})

plt.figure(figsize=(16, 12))

dataset = "reddit"

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
# plt.ylim(0.88, 0.98)
plt.xlabel('adding #edges (k) ')

no_retrain = load_data("{}_no.txt".format(dataset))
retrain30 = load_data("{}_30.txt".format(dataset))
retrain10 = load_data("{}_10.txt".format(dataset))
retrain5 = load_data("{}_5.txt".format(dataset))
retrain20 = load_data("{}_20.txt".format(dataset))
retrain50 = load_data("{}_50.txt".format(dataset))
iter_500 = 500

linewidth = 5

plt.plot(np.arange(len(no_retrain)), no_retrain, linewidth=linewidth, label="no retrain")
plt.plot(np.arange(len(retrain5)), retrain5, linewidth=linewidth, label="retrain every 5k")
plt.plot(np.arange(len(retrain10)), retrain10, linewidth=linewidth, label="retrain every 10k")
plt.plot(np.arange(len(retrain20)), retrain20, linewidth=linewidth, label="retrain every 20k")
plt.plot(np.arange(len(retrain30)), retrain30, linewidth=linewidth, label="retrain every 30k")
plt.plot(np.arange(len(retrain50)), retrain50, linewidth=linewidth, label="retrain every 50k")
# plt.xlim(0.85, 1.0)

plt.legend()

# plt.savefig("e2e_bert.svg", dpi=400, bbox_inches='tight')
plt.savefig("{}.png".format(dataset), dpi=400, bbox_inches='tight')