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

linewidth = 5
plt.title(dataset+"_degree")
def load_data(file):
    retrain = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            retrain.append(float(line))

    retrain = np.array(retrain) 

    return retrain

no_retrain = load_data("{}_no.txt".format(dataset))
# # retrain2 = load_data("{}_2.txt".format(dataset))
# # retrain6 = load_data("{}_6.txt".format(dataset))
# # retrain11 = load_data("{}_11.txt".format(dataset))
# # retrain21 = load_data("{}_21.txt".format(dataset))
# # retrain51 = load_data("{}_51.txt".format(dataset))
# # retrain2 = load_data("{}_2.txt".format(dataset))
# retrain6 = load_data("{}_5.txt".format(dataset))
# retrain11 = load_data("{}_10.txt".format(dataset))
# retrain21 = load_data("{}_20.txt".format(dataset))
# retrain31 = load_data("{}_30.txt".format(dataset))
# retrain51 = load_data("{}_50.txt".format(dataset))
no_retrain_avg = np.average(no_retrain)
# # retrain2_avg = np.average(retrain2)
# # print(retrain2_avg)
# retrain6_avg = np.average(retrain6)
# retrain11_avg = np.average(retrain11)
# retrain21_avg = np.average(retrain21)
# retrain31_avg = np.average(retrain31)
# retrain51_avg = np.average(retrain51)
# x = np.array([5, 10, 20, 30, 50])
# y = np.array([retrain6_avg, retrain11_avg, retrain21_avg, retrain31_avg, retrain51_avg])
# plt.plot(x, y, linewidth=linewidth, label="replay ratio = 1")
x = np.array([50, 30, 20, 10, 5])
# WIKI
y0 = [0.9372128561, 0.9406325595, 0.9462103564, 0.9499589222, 0.9499812548]
# WIKI with optimizer reset
# y1 = [0.9377699649, 0.9420297883, 0.9492591743, 0.9536993728, 0.9604802084]
# y2 = [0.9393416445, 0.9437706799, 0.9447926902, 0.9513899164, 0.9555025506]
# y3 = [0.9357152842, 0.9440252309, 0.9441685783, 0.9471832228, 0.9502027354]
# y4 = [0.9317848266, 0.9356779964, 0.9330054898, 0.9313954608, 0.939256955]
# WIKI using uniform
# y1 = [0.939317788, 0.9472429372, 0.9485351233, 0.9540912788, 0.9586445901]
# y2 = [0.9344845369, 0.942449245, 0.9481023196, 0.9507080008, 0.9548851508]
# y3 = [0.9340914712, 0.9440228166, 0.9425435784, 0.9454642073, 0.9500756867]
# y4 = [0.9300341052, 0.9352477044, 0.9320649371, 0.9320196328, 0.9366625135]
# # WIKI using LOSS
# y1 = [0.9381209894, 0.9466805205, 0.9468017224, 0.9534519088, 0.9590586771]
# y2 = [0.9354536683, 0.9421413249, 0.9439648047, 0.9487986603, 0.9525680731]
# y3 = [0.933479649, 0.9388646914, 0.9425015068, 0.9424390063, 0.9462359658]
# y4 = [0.9283644214, 0.9339612561, 0.933202162, 0.931559147, 0.9403000512]
# WIKI using DEGREE
y1 = [0.9385320182, 0.9451635385, 0.9464094222, 0.9561380271, 0.9584793886]
y2 = [0.9356484281, 0.943748202, 0.9448622145, 0.9505632967, 0.9537089988]
y3 = [0.9322362852, 0.9414035053, 0.9438502047, 0.9463067086, 0.9490714081]
y4 = [0.9298167481, 0.9361813128, 0.9316859737, 0.9333750301, 0.9415867447]
# MOOC with optimizer reset
# y1 = [0.6585522509, 0.6586995857, 0.6630724573, 0.6634534598, 0.6659212507]
# y2 = [0.6597825378, 0.6620917413, 0.6615614477, 0.6659833381, 0.6693825379]
# y3 = [0.6594982952, 0.6574196158, 0.6566554758, 0.6593854083, 0.6658508959]
# y4 = [0.6581822051, 0.6511822371, 0.6490324292, 0.6409949778, 0.6476071031]
# # MOOC using uniform
# y1 = [0.6580256391, 0.6566543196, 0.6620894608, 0.6632175399, 0.6665499233]
# y2 = [0.6576317998, 0.657590987, 0.6557479741, 0.6560899903, 0.6579741437]
# y3 = [0.6575014035, 0.6525267783, 0.6490239107, 0.6471580462, 0.64398593]
# y4 = [0.6567372507, 0.654834751, 0.6472927845, 0.6415313819, 0.64673103]
# MOOC using DEGREE
y2 = [0.656169472, 0.6578351241, 0.6575946904, 0.6561753291, 0.6535851259]
y3 = [0.657378181, 0.6523115819, 0.6505558701, 0.641449211, 0.6403866774]
y4 = [0.6541339424, 0.655314556, 0.6548153142, 0.6589513679, 0.6588454603]
# MOOC using LOSS
y2 = [0.6541339424, 0.655314556, 0.6548153142, 0.6589513679, 0.6588454603]
y3 = [0.6508916774, 0.6510954522, 0.650818787, 0.6497926315, 0.6501010959]
y4 = [0.6444204174, 0.6409804458, 0.6368663232, 0.6277772824, 0.6385720419]
# WIKI DATA
# y2 = [0.9330540196, 0.9316736323, 0.9382383579, 0.9470784184, 0.9497551646]
# y3 = [0.9275608499, 0.9324747436, 0.9301592745, 0.9270544306, 0.9385793581]
# y4 = [0.9364841262, 0.9405177975, 0.9363309849, 0.9426335953, 0.9481003786]
# MOOC DATA
# y0 = [0.6413316043, 0.6302047448, 0.643727914, 0.6443710574, 0.6601391029]
# y2 = [0.6540648877, 0.6556331394, 0.6586561942, 0.6632047736, 0.6701996005]
# y3 = [0.6412016762, 0.6337602869, 0.6290732136, 0.6148852069, 0.6369092979]
# y4 = [0.6456596868, 0.6527399919, 0.6520831922, 0.6549432243, 0.6598520666]
# plt.plot(x, y0, linewidth=linewidth, label="train from scratch")
plt.plot(x, y1, linewidth=linewidth, label="replay ratio = 1")
plt.plot(x, y2, linewidth=linewidth, label="replay ratio = 0.5")
plt.plot(x, y3, linewidth=linewidth, label="replay ratio = 0.2")
plt.plot(x, y4, linewidth=linewidth, label="replay ratio = 0")
# plt.ylim(0.605, 0.67)
plt.ylim(0.89, 0.97)
# plt.ylim(0.975, 0.985)
plt.xlabel("retrain interval (k)")
plt.ylabel("average precision")
plt.axhline(no_retrain_avg, linewidth=linewidth, color='r', linestyle='--', label="w/o retrain")
plt.legend()

plt.savefig("{}_pareto_degree.png".format(dataset), dpi=400, bbox_inches='tight')


