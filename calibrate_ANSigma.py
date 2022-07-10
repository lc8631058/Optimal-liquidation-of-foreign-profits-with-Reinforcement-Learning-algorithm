from utilities import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

#################################################################
########################### Offline calibration #######################################
#################################################################

# with open("./Data/data_FX_dayly.pkl", "rb") as f:
#     data = pickle.load(f)
# x = data['EUR' + 'USD' + '_Curncy'].to_numpy()
# t = np.linspace(0, x.shape[0], x.shape[0])
# xcoords = list(range(0, x.shape[0], 100))
# plt.figure()
# plt.plot(t, x)
# for xc in xcoords:
#     plt.axvline(x=xc, c='r')
# plt.show()

dt = 1.0
currency = 'CNY'
# StartIndex = [0, 20,40, 60,80]
StartIndex = [0]
Interval = [100]
len_= 250
test_for_OnlineLR_1 = True
# test_for_OnlineLR_1 = False
if test_for_OnlineLR_1:
    # load full data
    with open("./Data/data_FX_dayly.pkl", "rb") as f:
        data = pickle.load(f)

    for inte in Interval:
        for index in StartIndex:
            cny_ = {}
            A_list = []
            N_list = []
            Sigma_list = []
            x_0_List = []
            for i_ in range(index, 4975, inte):
                cny_['EUR' + currency + '_Curncy'] = data['EUR' + currency + '_Curncy'][i_:len_+i_]
                x_0 = np.log(data['EUR' + currency + '_Curncy'][i_])
                A, N, Sigma = calibration_fx(cny_, dt, if_log=True)
                A_list.append(A.item())
                N_list.append(N.item())
                Sigma_list.append(Sigma.item())
                x_0_List.append(x_0)

            # save serie and calibrated label
            with open("./data/ANSigma/{}_ANSigmaX0_Intval{}_StartIndex{}_len{}.pkl".format(currency, inte, index, len_), "wb") as f:
                pickle.dump((A_list, N_list, Sigma_list, x_0_List), f)

curncy_plot = 'USD'
StartIndex_plot = 0
Interval_plot = 100
T = 100
M = 2000
with open("./data/ANSigma/{}_ANSigmaX0_Intval{}_StartIndex{}_len{}.pkl".format(currency, Interval_plot, StartIndex_plot, len_), "rb") as f:
    A_list, N_list, Sigma_list, x_0_List = pickle.load(f)
t = np.linspace(0, T, T + 1)
Path('./figs/USD_ANSigma_Samples/CaliStartFromIndex_{}_Intval{}_len{}_T{}'.format(StartIndex_plot, Interval_plot, len_, T)).mkdir(parents=True, exist_ok=True)
for i in range(len(A_list)):
    # X = MC_simulation(0, M, T, x_0_List[i], -A_list[i]/2, -N_list[i]/2, Sigma_list[i])
    X = MC_simulation(0, M, T, x_0_List[i], A_list[i], N_list[i], Sigma_list[i])
    x_min, x_max = np.min(X), np.max(X)
    # mean_level = np.exp(N / (-A))
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('FX rate')
    plt.title('i={}'.format(i))
    for m in range(0, M//2, 10):
        plt.plot(t, np.exp(X[M// 2 + m]))
        # plt.show()
    add = './figs/USD_ANSigma_Samples/CaliStartFromIndex_{}_Intval{}_len{}_T{}/USD_{}th.jpg'.format(StartIndex_plot, Interval_plot, len_, T, i)
    plt.savefig(add)
    # plt.legend()
    # plt.show()
