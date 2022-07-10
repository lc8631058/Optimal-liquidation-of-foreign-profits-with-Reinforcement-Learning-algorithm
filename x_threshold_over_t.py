import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utilities import *

currency = 'USD'
expand_steps_ForTrain = 200 # the steps you wanna decoder to expand
A_list, N_list, Sigma_list, x_0_List, idx_list = [], [], [], [], []
label = []
for i in range(2, 96):
    with open('./experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_AvgANSigma.pkl'.format(
            expand_steps_ForTrain, currency, i, currency), 'rb') as f:
        A, N, SIGMA, START_PRICE, LABEL = pkl.load(f)
    idx_list.append(i)
    A_list.append(A)
    N_list.append(N)
    Sigma_list.append(SIGMA)
    x_0_List.append(START_PRICE)
    label.append(LABEL)

trend = 'flat'
# when i == 0 in calibration
# # Downtrend
if trend=='down':
    A = -0.02038755
    N = -0.00205892
    SIGMA = 0.0074969
    START_PRICE = 0.030141156911986808

    ## Real
    # i_ = 19
    # A = A_list[i_]
    # N = N_list[i_]
    # SIGMA = Sigma_list[i_]
    # START_PRICE = x_0_List[i_]

# # when i == 46
# # Flat trend
elif trend=='flat':
    A = -0.07930878
    N = 0.01341533
    SIGMA = 0.0038368
    START_PRICE = 0.1655991806495659

    # Real
    # i_ = 2
    # A = A_list[i_]
    # N = N_list[i_]
    # SIGMA = Sigma_list[i_]
    # START_PRICE = x_0_List[i_]

# when i == 20
# Up trend
elif trend=='up':
    A = -0.05549867
    N = 0.02086743
    SIGMA = 0.00464528
    START_PRICE = 0.30814624558413484

    # # Real
    # i_ = 0
    # A = A_list[i_]
    # N = N_list[i_]
    # SIGMA = Sigma_list[i_]
    # START_PRICE = x_0_List[i_]

N_pqc = 10
N_x = 600
v_0 = 100
N_v = 100
Delta_v = v_0 / N_v

M = 2000
T = 100
v_min = 10
v_max = 25

X = MC_simulation(0, M, T, START_PRICE, A, N, SIGMA)
X_train, X_test = X[:1000, :], X[1000:, :]
x_min, x_max = np.min(X), np.max(X)
Delta_x = (x_max - x_min) / N_x
# read estimated SDP
# with open('./data/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}.pkl'.format(
#         M // 2, N_pqc, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE), 'rb') as f:
#     f_star, f_history, x_thres = pkl.load(f)
# read SDP
# address_down = './data/Nx600_Nv100_T100_v0100_vmin10_vmax25_xmin-0.24770_xmax0.09861_ANSIGMA-0.02039--0.00206-0.00750.pkl'
# address_flat = './data/Nx600_Nv100_T100_v0100_vmin10_vmax25_xmin0.12193_xmax0.21062_ANSIGMA-0.07931-0.01342-0.00384.pkl'
address_up = './data/Nx600_Nv100_T100_v0100_vmin10_vmax25_xmin0.28814_xmax0.43437_ANSIGMA-0.05550-0.02087-0.00465.pkl'
with open(address_up, 'rb') as f:
    F_STAR, A_STAR = pkl.load(f)
# with open('./data/SDP/Real_{}/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_{}th.pkl'.format(
#         currency, N_x, N_v, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, idx_list[i_]), 'rb') as f:
#     F_STAR, A_STAR = pkl.load(f)
# with open('./data/Exp_Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}.pkl'.format(N_x, N_v, T, v_0, v_min, v_max), 'rb') as f:
#     F_STAR, A_STAR = pkl.load(f)

def first_nonzero(arr, axis, invalid_val=-1):
    # find the index of first nonzero value in array given axis,
    # when there's no nonzero value, return invalid_val
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    # find the index of last nonzero value in array given axis and +1 (found index + 1),
    # when there's no nonzero value, return invalid_val
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val + 1, invalid_val)

# count the x_threshold for each
x_thres = np.zeros([T], dtype=np.float32)

for t in range(T):
    tmp = last_nonzero(A_STAR[:, :, t], 0, invalid_val=0)
    if tmp[np.nonzero(tmp)].size != 0:
        x_thres[t] = x_min +  np.max(tmp[np.nonzero(tmp)]) * Delta_x
    else:
        x_thres[t] = x_min
coef = np.polyfit(np.arange(T), x_thres, 2)

def rebuildFunc(x, coef):
    return coef[0] * x**2 + coef[1] * x + coef[2]

rebuild = rebuildFunc(np.arange(T), coef)

mean_level = N / (-A)
fig = plt.figure()
t1 = np.linspace(0, T, T)
t2 = np.linspace(0, T, T + 1)
plt.xlabel('t', fontsize=16)
plt.ylabel('minimum of x when a*(x,v,t)=0 for all v', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=11)
# for i in range(X_test.shape[0]):
#     plt.plot(t2, np.exp(X_test[i]))
plt.plot(t1, np.exp(x_thres), c='b', label='x_thres')
# plt.plot(t1, np.exp(Esti_x_thres[:-1]), c='r', label='Esti_x_thres', linewidth=3.0)
plt.plot(t2, [np.exp(mean_level)] * (T + 1), label='mean level')
plt.plot(t2, [np.exp((x_min + x_max)/2)] * (T + 1), label='middle of min and max')
# plt.plot(t, rebuild, 'r', label='rebuild')
plt.legend(loc='best', prop={'size': 12})

# plt.savefig('./3d_plots/figures/Real_{}/x_thres_{}.pdf'.format(trend, trend))
plt.savefig('./3d_plots/figures/{}/x_thres_{}.pdf'.format(trend, trend))
# plt.show()
a
## plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# X = np.arange(0, 201 * Delta_v, Delta_v)
# # X = np.arange(x_min, x_min + 101 * Delta_x, Delta_x)
# ax.set_xlabel('$v$', fontsize=15)
# Y = np.arange(0, T + 1, 1)
# ax.set_ylabel('$t$', fontsize=15)
# X, Y = np.meshgrid(X, Y)
# ax.set_zlabel('$x$', fontsize=15, rotation=0)
#
# ax.view_init(elev=90., azim=370 - 90 - 90)
#
# surf = ax.plot_surface(X, Y, x_thres.T, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=True)
# plt.show()
