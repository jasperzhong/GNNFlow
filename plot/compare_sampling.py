import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline, make_interp_spline

plt.rcParams.update({'font.size': 26, 'font.family': 'Myriad Pro'})

plt.figure(figsize=(12, 9))

dataset = "mooc"

linewidth = 5
plt.title(dataset+"replay ratio 0.5")
def load_data(file):
    retrain = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            retrain.append(float(line))

    retrain = np.array(retrain) 

    return retrain

no_retrain = load_data("{}_no.txt".format(dataset))
no_retrain_avg = np.average(no_retrain)
# # WIKI replay ratio = 0.5
# y_our = [0.9393416445, 0.9437706799, 0.9447926902, 0.9513899164, 0.9555025506]
# y_uniform = [0.9344845369, 0.942449245, 0.9481023196, 0.9507080008, 0.9548851508]
# y_loss = [0.9354536683, 0.9421413249, 0.9439648047, 0.9487986603, 0.9525680731]
# y_degree = [0.9356484281, 0.943748202, 0.9448622145, 0.9505632967, 0.9537089988]
# WIKI replay ratio = 0.2
# y_our = [0.9357152842, 0.9440252309, 0.9441685783, 0.9471832228, 0.9502027354]
# y_uniform = [0.9340914712, 0.9440228166, 0.9425435784, 0.9454642073, 0.9500756867]
# y_loss = [0.933479649, 0.9388646914, 0.9425015068, 0.9424390063, 0.9462359658]
# y_degree = [0.9322362852, 0.9414035053, 0.9438502047, 0.9463067086, 0.9490714081]
# # MOOC replay ratio = 0.2
# y_our = [0.6594982952, 0.6574196158, 0.6566554758, 0.6593854083, 0.6658508959]
# y_uniform = [0.6575014035, 0.6525267783, 0.6490239107, 0.6471580462, 0.64398593]
# y_loss = [0.6541339424, 0.655314556, 0.6548153142, 0.6589513679, 0.6588454603]
# y_degree = [0.657378181, 0.6523115819, 0.6505558701, 0.641449211, 0.6403866774]
# # MOOC replay ratio = 0.5
y_our = [0.6597825378, 0.6620917413, 0.6615614477, 0.6659833381, 0.6693825379]
y_uniform = [0.6576317998, 0.657590987, 0.6557479741, 0.6560899903, 0.6579741437]
y_degree = [0.656169472, 0.6578351241, 0.6575946904, 0.6561753291, 0.6535851259]
y_loss = [0.6541339424, 0.655314556, 0.6548153142, 0.6589513679, 0.6588454603]

x = np.array([50, 30, 20, 10, 5])

plt.plot(x, y_our, linewidth=linewidth, label="our method")
plt.plot(x, y_uniform, linewidth=linewidth, label="uniform sampling")
plt.plot(x, y_degree, linewidth=linewidth, label="degree based sampling")
plt.plot(x, y_loss, linewidth=linewidth, label="loss based sampling")

plt.xlabel("retrain interval (k)")
plt.ylabel("average precision")
plt.axhline(no_retrain_avg, linewidth=linewidth, color='r', linestyle='--', label="w/o retrain")
plt.legend()

plt.savefig("{}_replay_{}.png".format(dataset, 0.5), dpi=400, bbox_inches='tight')