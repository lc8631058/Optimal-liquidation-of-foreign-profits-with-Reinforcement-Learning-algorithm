import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle as pkl
import numpy as np
import matplotlib.animation as animation
import os
import torch
from Environment import *
from DQN_Agent import *
# from PPO.PPO_Agent import PPO
from PPO_Agent import *
import time
from utilities import *

def MC_simulation(seed, M, T, START_PRICE, A, N, SIGMA):
    np.random.seed(seed)
    DELTA_W = np.random.normal(loc=0., scale=1., size=(M, T))
    X = np.zeros([M, T + 1], dtype=np.float32)
    for m in range(M):
        y = START_PRICE
        X[m, 0] = START_PRICE
        for t in range(1, T + 1):
            y = y + A * y + N + SIGMA * DELTA_W[m, t - 1]
            X[m, t] = y
    return X

def function(t, p, q, c):
    # t is a vector
    return p * (t * t) + q * t + c

def pqc_condition(p, q, c):
    if p < 0 or q < 0 or c < x_min or p * (T**2) + q * T + c > x_max:
        return False
    else:
        return True

def generate_pqc(N, T, x_min, x_max):
    R = []
    p_min, p_max = 0, (x_max - x_min) / (T ** 2)
    q_min, q_max = 0, (x_max - x_min) / T
    c_min, c_max = x_min, x_max
    Delta_p = (x_max - x_min) / (N * (T ** 2))
    Delta_q = (x_max - x_min) / (N * T)
    Delta_c = (x_max - x_min) / N

    for i in range(N + 1):
        p_i = p_min + i * Delta_p
        for j in range(N + 1):
            q_j = q_min + j * Delta_q
            for k in range(N + 1):
                c_k = c_min + k * Delta_c
                if pqc_condition(p_i, q_j, c_k):
                    R.append((p_i, q_j, c_k))
    return R
#
def TWAP(X, v_min, T, interval, remain, v_0):
    f = np.zeros([X.shape[0]], dtype=np.float32)
    v_traj = np.zeros([X.shape[0], T + 2], dtype=np.float32)
    v_traj[:, 0] = v_0
    start = time.time()
    TWAP_strategy = []
    for m in range(X.shape[0]):
        v = v_0
        cnt = 0
        TWAP_action = []
        for t in range(T + 1):
            if t == 0 or (cnt > 0 and cnt % interval == 0):
                a = v_min
                cnt = 0
            else:
                a = 0.
                cnt += 1
            f[m] += a / np.exp(X[m, t])
            v = v - a
            v_traj[m, t + 1] = v
            TWAP_action.append(a)
        # sell remain at T
        f[m] += remain / np.exp(X[m, T])
        TWAP_strategy.append(TWAP_action)
    end = time.time()
    print('TWAP time cost: {}'.format(end - start))
    return f, np.mean(f), np.std(f), np.median(f), v_traj, TWAP_strategy

def estimated_a_star(x_thres, x, v, t, T, v_min, v_max):
    """
    return estimated a*
    :param x_thres: estimated optimal threshold of x over t
    """
    if t == T:
        return v
    else:
        return np.heaviside(x_thres[t] - x, 1) * \
               (min(v_max, max(0, v - v_min) + v_min * np.heaviside(v - v_min, 1) * np.heaviside(v_max - v, 1)))

def EOLS_strategy(X, T, x_thres, v_min, v_max, v_0):
    """

    :param X:
    :param T:
    :param x_thres:
    :param v_min:
    :param v_max:
    :return:
    """
    f = np.zeros([X.shape[0]], dtype=np.float32)
    v_traj = np.zeros([X.shape[0], T + 2], dtype=np.float32)
    v_traj[:, 0] = v_0
    EOLS_strategy = []
    for m in range(X.shape[0]):
        v = v_0
        EOLS_action = []
        for t in range(T + 1):
            a = estimated_a_star(x_thres, X[m, t], v, t, T, v_min, v_max)
            f[m] += a / np.exp(X[m, t])
            v = v - a
            v_traj[m, t + 1] = v
            EOLS_action.append(a)
        EOLS_strategy.append(EOLS_action)
    return f, np.mean(f), np.std(f), np.median(f), v_traj, EOLS_strategy

def SDP_strategy(X, T, A_STAR, x_min, x_max, N_x, v_0, Delta_v):
    """

    :param X:
    :param T:
    :param A_STAR: strategy from SDP
    :return:
    """
    Delta_x = (x_max - x_min) / N_x
    f = np.zeros([X.shape[0]], dtype=np.float32)
    v_traj = np.zeros([X.shape[0], T + 2], dtype=np.float32)
    v_traj[:, 0] = v_0
    SDP_strategy = []
    for m in range(X.shape[0]):
        v = v_0
        SDP_action = []
        for t in range(T + 1):
            if int((X[m, t]-x_min)//Delta_x) < 0:
                a = A_STAR[0, int(v//Delta_v), t]
            else:
                a = A_STAR[min(int((X[m, t]-x_min)//Delta_x), A_STAR.shape[0]-1), int(v//Delta_v), t]
            f[m] += a / np.exp(X[m, t])
            v = v - a
            v_traj[m, t + 1] = v
            SDP_action.append(a)
        SDP_strategy.append(SDP_action)
    return f, np.mean(f), np.std(f), np.median(f), v_traj, SDP_strategy

currency= 'USD'
StartIndex = 0
Interval = 100
len_= 250
# with open("./data/ANSigma/{}_ANSigmaX0_Intval{}_StartIndex{}_len{}.pkl".format(currency, Interval, StartIndex, len_), "rb") as f:
#     A_list, N_list, Sigma_list, x_0_List = pkl.load(f)
trend = 'uptrend'
# trend = 'sideways'
# trend = 'downtrend'
# trend = 'fluctuation'
# trend='-'
# i = 1

N_x = 100
v_0 = 100
N_v = 100
N_pqc = 10
Delta_v = v_0 / N_v
v_min = 10
v_max = 25

T = 100
M = 2000

# plot DQN, PPO, TWAP, EOLS's strategy
#============================= DQN =============================
# DQN params
STATE_SIZE = 3   # (x, v, t)
ACTION_SIZE = 17 # [0, V_MIN, V_MIN + 1, ..., V_MAX]
FC1_UNITS, FC2_UNITS, FC3_UNITS, FC4_UNITS, FC5_UNITS = [64] * 5
LR = 0.00025
MOMENTUM = 0.95
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 32
CURVE_BATCH_SIZE = 1
UPDATE_EVERY = 10000
TAU = 0.1              # for soft update of target network
EPS_START = 0.95         # start value for epsilon-greedy decay
EPS_END = 0.          # end value for epsilon-greedy decay
EPS_DECAY = 50000         # decay every
RANDOM_SEED = 0

# real FX rate
expand_steps_ForTrain = 200  # the steps you wanna decoder to expand
A_list, N_list, Sigma_list, x_0_List, idx_list = [], [], [], [], []
label, pred_results = [], []
for i in range(2, 96):
    with open('./experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_AvgANSigma.pkl'.format(
            expand_steps_ForTrain, currency, i, currency), 'rb') as f:
        A, N, SIGMA, START_PRICE, LABEL, PRED_RESULTS = pkl.load(f)
    idx_list.append(i)
    A_list.append(A)
    N_list.append(N)
    Sigma_list.append(SIGMA)
    x_0_List.append(START_PRICE)
    label.append(np.concatenate([np.exp(START_PRICE.reshape(1, )), LABEL.squeeze()], 0))
    pred_results.append(np.concatenate([np.exp(START_PRICE.reshape(1,)), np.mean(PRED_RESULTS.squeeze(), 0)], 0))
label = np.array(np.log(label))
# label = np.concatenate([label, np.expand_dims(label[:, -1], 1)], 1)

# for i in range(37, len(A_list)):
xx = 13
for i in range(xx, xx+1):
    A, N, SIGMA, START_PRICE, LABEL = A_list[i-2], N_list[i-2], Sigma_list[i-2], x_0_List[i-2], label[i-2]
    # A, N, SIGMA, START_PRICE, LABEL = A_list[i-1], N_list[i-1], Sigma_list[i-1], x_0_List[i-1], label[i-1]
    ith_traj = 0
    X = MC_simulation(0, M, T, START_PRICE, A, N, SIGMA)
    X_train, X_test = X[:1000, :], X[1000:, :]
    x_min, x_max = np.min(X), np.max(X)
    Delta_x = (x_max - x_min) / N_x

    # plt.figure()
    # t = np.linspace(0, T, T +1)
    # for i in range(X_train.shape[0]):
    #     plt.plot(t, X_train[i])
    # plt.show()

    # model_path = './model/DQN/{}/DQN_Model_{}_State{}Action{}_LR{}_5FC{}_EPS_DECAY{}_Update{}_Tau{}_{}th.pkl'.format(
    #     currency, currency, STATE_SIZE, ACTION_SIZE, LR, FC1_UNITS, EPS_DECAY, UPDATE_EVERY, TAU, i)
    model_path = './model/DQN/Real_{}/DQN_Model_{}_State{}Action{}_LR{}_5FC{}_EPS_DECAY{}_Update{}_Tau{}_{}th.pkl'.format(
        currency, currency, STATE_SIZE, ACTION_SIZE, LR, FC1_UNITS, EPS_DECAY, UPDATE_EVERY, TAU, i)

    checkpoint = torch.load(model_path)

    # ----------------------------- Market Env ----------------------------- #
    Market = MarketEnvironment(random=RANDOM_SEED,
                               liquid_time=T,
                               total_vol=v_0,
                               v_min=v_min,
                               v_max=v_max)
    # ----------------------------- DQN Agent ----------------------------- #
    DQN_Agent = Agent(STATE_SIZE, ACTION_SIZE,
                      FC1_UNITS, FC2_UNITS, FC3_UNITS, FC4_UNITS, FC5_UNITS,
                      LR, MOMENTUM, BUFFER_SIZE, BATCH_SIZE, UPDATE_EVERY, TAU,
                      EPS_START, EPS_END, EPS_DECAY, RANDOM_SEED)

    DQN_Agent.qnetwork_policy.load_state_dict(checkpoint['DQN_Policy_state_dict'])
    DQN_Agent.qnetwork_target.load_state_dict(checkpoint['DQN_Target_state_dict'])
    DQN_Agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Market.run_newEpisode(X_test)  # run one episode
    Market.run_newEpisode(np.expand_dims(LABEL, 0))  # test real FX rate

    state = Market.get_initial()  # get the initial state
    Market.start_transactions()  # start the transaction

    DQN_strategy = []
    DQN_remain_v = []
    for t in range(T + 1):
        # select an action
        action = DQN_Agent.act(state, ifTest=True)
        if t == T-1:
            a = 1
        # NN output space = [0, 1, ..., 16], the action space = [0, 10, ..., 25]
        actual_action = action[0][0] + v_min - 1 if action[0][0] != 0 else 0
        if actual_action < state[0, 1] and state[0, 1] >= v_min:
            DQN_strategy.append(actual_action)
        elif int(state[0, 2]) == 0 and state[0, 1] < v_min:
            DQN_strategy.append(int(state[0, 1]))
        else:
            DQN_strategy.append(0)
        DQN_remain_v.append(state[0, 1])

        # Market.action_list.append(action)
        next_state, reward, done, _ = Market.step(action, state)
        # next_state, reward, PL, done, _ = Market.step(action, state)
        Market.score += reward
        # Market.PL_score += PL
        # next_state to state
        state = next_state
        # transaction finished
        # if done.all():
        #     break
    DQN_PL = Market.score
    # DQN_Optimality, _ = captured_percentage_index(np.expand_dims(X_test[ith_traj, :], 0), v_0, DQN_PL)
    DQN_Optimality, _ = captured_percentage_index(np.expand_dims(LABEL, 0), v_0, DQN_PL)


    #============================= EOLS =============================
    # read estimated EOLS
    N_pqc = 10
    # address1 = './data/Esti-SDP/{}/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}_{}th.pkl'.format(
    #             currency, M // 2, N_pqc, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE,
    #             i)
    address1 = './data/Esti-SDP/Real_{}/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}_{}th.pkl'.format(
        currency, M // 2, N_pqc, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE,
        i)  # read estimated SDP
    with open(address1, 'rb') as f:
        f_star, f_history, x_thres = pkl.load(f)

    # estimation
    # EOLS_PL, _, _, _, EOLS_v_traj, EOLS_strategy = EOLS_strategy(np.expand_dims(X_test[ith_traj, :], 0), T, x_thres, v_min,
    #                                                                                 v_max, v_0)
    # EOLS_Optimality, _ = captured_percentage_index(np.expand_dims(X_test[ith_traj, :], 0), v_0, EOLS_PL)

    EOLS_PL, _, _, _, EOLS_v_traj, EOLS_strategy = EOLS_strategy(np.expand_dims(LABEL, 0), T, x_thres,
                                                                 v_min,
                                                                 v_max, v_0)
    EOLS_Optimality, _ = captured_percentage_index(np.expand_dims(LABEL, 0), v_0, EOLS_PL)

    #============================= SDP =============================
    # read SDP
    # address2 = './data/SDP/{}/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_{}th.pkl'.format(
    #     currency, N_x, N_v, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, i)
    address2 = './data/SDP/Real_{}/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_{}th.pkl'.format(
        currency, N_x, N_v, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, i)

    with open(address2, 'rb') as f:
        F_STAR, A_STAR = pkl.load(f)
    # SDP
    SDP_PL, _, _, _, SDP_v_traj, SDP_strategy = SDP_strategy(np.expand_dims(LABEL, 0), T, A_STAR, x_min, x_max, N_x, v_0,
                                                                          Delta_v)
    SDP_Optimality, _ = captured_percentage_index(np.expand_dims(LABEL, 0), v_0, SDP_PL)

    #============================= TWAP =============================
    n = v_0 // v_min
    remain = v_0 - n * v_min
    interval = T // n
    # Naive_PL, Naive_PL_mean, Naive_PL_std, Naive_PL_median, Naive_v_traj = stat_Naive(X_test, v_min, T, interval, remain, v_0)
    TWAP_PL, TWAP_PL_mean, TWAP_PL_std, TWAP_PL_median, TWAP_v_traj, TWAP_strategy = TWAP(np.expand_dims(LABEL, 0), v_min, T, interval, remain, v_0)
    TWAP_Optimality, _ = captured_percentage_index(np.expand_dims(LABEL, 0), v_0, TWAP_PL)







    #=====================================================================
    #============================= Visualize =============================
    #=====================================================================


    #============================= DQN =============================

    # Creating dataset
    MEAN, STD = np.mean(np.exp(X), 0), np.std(np.exp(X), 0)
    t = np.linspace(0, T, T + 1)
    t2 = np.linspace(0, T + 1, T + 2)
    # Creating plot with dataset_1
    fig, axs = plt.subplots()

    color = 'tab:cyan'
    axs.set_xlabel('time', fontdict={'size': 14})
    axs.set_ylabel('FX rate (label)', color=color, fontdict={'size': 14})

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # axs.scatter(t, DQN_strategy, color=color, s=3)

    # axs.plot(t, np.exp(X_test[ith_traj, :]), color=color)
    axs.plot(t, np.exp(LABEL), color=color)
    axs.plot(t, np.exp(np.mean(X, 0)), color='blue', label='forecasted FX rate')
    # axs.plot(t, pred_results[i-1], color='blue', label='forecasted FX rate')
    axs.fill_between(t, (MEAN + 2*STD), (MEAN - 2*STD), color='b', alpha=.1)

    axs.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = axs.twinx()

    color = 'tab:green'
    ax2.set_ylabel('liquidation trajectory', color=color, fontdict={'size': 13})
    ax2.plot(t2, DQN_remain_v + [0], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.yticks(fontsize=12)

    # captured FX
    captured_FX = v_0 / DQN_PL
    axs.plot(t2, [captured_FX[0,0]] * (T + 2), color='tab:red', label='captured FX rate', linewidth=2.0)

    # Adding title
    ss = 'O_{liq}'
    plt.title('DQN, EUR/{}, {}, ${}$: {:.3}, index: {}'.format(currency, trend, ss, DQN_Optimality, i), fontdict={'size': 14})
    axs.legend(loc='best', prop={'size': 14})
    # Show plot
    # plt.show()
    # plt.savefig('./3d_plots/figures/DQN/DQN_strategy_{}_{}_{}th.pdf'.format(currency, trend, i), bbox_inches='tight')
    plt.savefig('./3d_plots/figures/DQN/Real_DQN_strategy_{}_{}_{}th.pdf'.format(currency, trend, i), bbox_inches='tight')




    #============================= EOLS =============================

    # Creating plot with dataset_1
    fig, axs = plt.subplots()

    color = 'tab:cyan'
    axs.set_xlabel('time', fontdict={'size': 14})
    axs.set_ylabel('FX rate (label)', color=color, fontdict={'size': 14})

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # axs.scatter(t, DQN_strategy, color=color, s=3)
    # axs.plot(t, np.exp(X_test[ith_traj, :]), color=color)
    axs.plot(t, np.exp(LABEL), color=color)
    axs.plot(t, np.exp(np.mean(X, 0)), color='blue', label='forecasted FX rate')
    axs.fill_between(t, (MEAN + 2*STD), (MEAN - 2*STD), color='b', alpha=.1)

    axs.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = axs.twinx()

    color = 'tab:green'
    ax2.set_ylabel('liquidation trajectory', color=color, fontdict={'size': 13})
    ax2.plot(t2, EOLS_v_traj[0, :] + [0], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.yticks(fontsize=12)

    # captured FX
    captured_FX = v_0 / EOLS_PL
    axs.plot(t2, [captured_FX[0]] * (T + 2), color='tab:red', label='captured FX rate', linewidth=2.0)

    # Adding title
    # plt.title('EOLS, EUR/{}, {}, ${}$: {:.3}, index: {}'.format(currency, trend, ss, EOLS_Optimality, i), fontdict={'size': 14})
    plt.title('EOLS + RegPred, EUR/{}, {}, ${}$: {:.3}, index: {}'.format(currency, trend, ss, EOLS_Optimality, i), fontdict={'size': 14})
    axs.legend(loc='best', prop={'size': 14})
    # Show plot
    # plt.show()
    # plt.savefig('./3d_plots/figures/EOLS/EOLS_strategy_{}_{}_{}th.pdf'.format(currency, trend, i), bbox_inches='tight')
    plt.savefig('./3d_plots/figures/EOLS/Real_EOLS_strategy_{}_{}_{}th.pdf'.format(currency, trend, i), bbox_inches='tight')

    #============================= SDP =============================

    # Creating plot with dataset_1
    fig, axs = plt.subplots()

    color = 'tab:cyan'
    axs.set_xlabel('time', fontdict={'size': 14})
    axs.set_ylabel('FX rate (label)', color=color, fontdict={'size': 14})

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # axs.scatter(t, DQN_strategy, color=color, s=3)
    # axs.plot(t, np.exp(X_test[ith_traj, :]), color=color)
    axs.plot(t, np.exp(LABEL), color=color)
    axs.plot(t, np.exp(np.mean(X, 0)), color='blue', label='forecasted FX rate')
    axs.fill_between(t, (MEAN + 2*STD), (MEAN - 2*STD), color='b', alpha=.1)

    axs.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = axs.twinx()

    color = 'tab:green'
    ax2.set_ylabel('liquidation trajectory', color=color, fontdict={'size': 13})
    ax2.plot(t2, SDP_v_traj[0, :] + [0], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.yticks(fontsize=12)

    # captured FX
    captured_FX = v_0 / SDP_PL
    axs.plot(t2, [captured_FX[0]] * (T + 2), color='tab:red', label='captured FX rate', linewidth=2.0)

    # Adding title
    plt.title('SDP, EUR/{}, {}, ${}$: {:.3}, index: {}'.format(currency, trend, ss, SDP_Optimality, i), fontdict={'size': 14})
    axs.legend(loc='best', prop={'size': 14})
    # Show plot
    # plt.show()
    # plt.savefig('./3d_plots/figures/SDP/SDP_strategy_{}_{}_{}th.pdf'.format(currency, trend, i), bbox_inches='tight')
    plt.savefig('./3d_plots/figures/SDP/Real_SDP_strategy_{}_{}_{}th.pdf'.format(currency, trend, i), bbox_inches='tight')


    #============================= TWAP =============================

    # Creating plot with dataset_1
    fig, axs = plt.subplots()

    color = 'tab:cyan'
    axs.set_xlabel('time', fontdict={'size': 14})
    axs.set_ylabel('FX rate (label)', color=color, fontdict={'size': 14})

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # axs.scatter(t, DQN_strategy, color=color, s=3)
    # axs.plot(t, np.exp(X_test[ith_traj, :]), color=color)
    axs.plot(t, np.exp(LABEL), color=color)
    axs.plot(t, np.exp(np.mean(X, 0)), color='blue', label='forecasted FX rate')
    axs.fill_between(t, (MEAN + 2*STD), (MEAN - 2*STD), color='b', alpha=.1)

    axs.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = axs.twinx()

    color = 'tab:green'
    ax2.set_ylabel('liquidation trajectory', color=color, fontdict={'size': 13})
    ax2.plot(t2, TWAP_v_traj[0, :] + [0], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.yticks(fontsize=12)

    # captured FX
    captured_FX = v_0 / TWAP_PL
    axs.plot(t2, [captured_FX[0]] * (T + 2), color='tab:red', label='captured FX rate', linewidth=2.0)

    # Adding title
    plt.title('TWAP, EUR/{}, {}, ${}$: {:.3}, index: {}'.format(currency, trend, ss, TWAP_Optimality, i), fontdict={'size': 14})
    axs.legend(loc='best', prop={'size': 14})
    # Show plot
    # plt.show()
    # plt.savefig('./3d_plots/figures/TWAP/TWAP_strategy_{}_{}_{}th.pdf'.format(currency, trend, i), bbox_inches='tight')
    plt.savefig('./3d_plots/figures/TWAP/Real_TWAP_strategy_{}_{}_{}th.pdf'.format(currency, trend, i), bbox_inches='tight')

