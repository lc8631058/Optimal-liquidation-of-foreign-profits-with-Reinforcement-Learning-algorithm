import numpy as np
import pickle as pkl
from utilities import *
import matplotlib.pyplot as plt
import time
from Environment import *
from DQN_Agent import *
import torch

# X = MC_simulation(0, M, T, START_PRICE, A, N, SIGMA)
# REWARD = []
# v_remain = 100
# for m in range(X.shape[0]):
#     reward = 0
#     for t in range(100 + 1):
#         if actions[t] != 0 and v_remain > 0:
#             reward += (actions[t] + 9) / np.exp(X[m, t])
#     REWARD.append(reward)

N_x = 100
v_0 = 100
N_v = 100
N_pqc = 10
Delta_v = v_0 / N_v
v_min = 10
v_max = 25

T = 100
M = 2000

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

####################################################################################
#                        Performance over entire dataset
####################################################################################
entire_test = True
# entire_test = False
if entire_test:
    currency= 'CNY'
    StartIndex = 0
    Interval = 100
    len_= 250
    # with open("./data/ANSigma/{}_ANSigmaX0_Intval{}_StartIndex{}_len{}.pkl".format(currency, Interval, StartIndex, len_), "rb") as f:
    #     A_list, N_list, Sigma_list, x_0_List = pkl.load(f)

    expand_steps_ForTrain = 200  # the steps you wanna decoder to expand

    # real FX rate
    A_list, N_list, Sigma_list, x_0_List, idx_list = [], [], [], [], []
    label = []
    for i in range(1, 96):
        with open('./experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_AvgANSigma.pkl'.format(
                expand_steps_ForTrain, currency, i, currency), 'rb') as f:
            A, N, SIGMA, START_PRICE, LABEL, PRED_RESULTS = pkl.load(f)
        idx_list.append(i)
        A_list.append(A)
        N_list.append(N)
        Sigma_list.append(SIGMA)
        x_0_List.append(START_PRICE)
        label.append(LABEL)
    label = np.array(np.log(label)).squeeze()
    label = np.concatenate([label, np.expand_dims(label[:, -1], 1)], 1)
    # SDP_PL_MEAN, SDP_PL_STD, SDP_PL_MED = [], [], []
    # SDP_Del_PL_MEAN, SDP_Del_PL_STD, SDP_Del_PL_MED, SDP_P_PL, SDP_GLR = [], [], [], [], []
    # EstiSDP_PL_MEAN, EstiSDP_PL_STD, EstiSDP_PL_MED = [], [], []
    # EstiSDP_Del_PL_MEAN, EstiSDP_Del_PL_STD, EstiSDP_Del_PL_MED, EstiSDP_P_PL, EstiSDP_GLR = [], [], [], [], []
    # Naive_PL_MEAN, Naive_PL_STD, Naive_PL_MED = [], [], []
    CPI_DQN_list, CPI_Esti_list, CPI_SDP_list, CPI_Naive_list = [], [], [], [] # captured percentage error
    # for i, (A, N, SIGMA, START_PRICE) in enumerate(zip(A_list, N_list, Sigma_list, x_0_List)):
    for i, A, N, SIGMA, START_PRICE in zip(idx_list, A_list, N_list, Sigma_list, x_0_List):
    #     if i == 20:
    #         a = 1
        X = MC_simulation(0, M, T, START_PRICE, A, N, SIGMA)
        X_train, X_test = X[:1000, :], X[1000:, :]
        x_min, x_max = np.min(X), np.max(X)
        Delta_x = (x_max - x_min) / N_x
        # LABEL = np.expand_dims(label[i - 2], 0)

        # real FX rate
        address1 = './data/Esti-SDP/Real_{}/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}_{}th.pkl'.format(
            currency, M // 2, N_pqc, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE,
            i)  # read estimated SDP
        with open(address1, 'rb') as f:
            f_star, f_history, x_thres = pkl.load(f)
        # read SDP
        address2 = './data/SDP/Real_{}/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_{}th.pkl'.format(
            currency, N_x, N_v, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, i)
        with open(address2, 'rb') as f:
            F_STAR, A_STAR = pkl.load(f)

        # address1 = './data/Esti-SDP/{}/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}_{}th.pkl'.format(
        #     currency, M // 2, N_pqc, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE,
        #     i)  # read estimated SDP
        # with open(address1, 'rb') as f:
        #     f_star, f_history, x_thres = pkl.load(f)
        # # read SDP
        # address2 = './data/SDP/{}/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_{}th.pkl'.format(
        #     currency, N_x, N_v, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, i)
        # with open(address2, 'rb') as f:
        #     F_STAR, A_STAR = pkl.load(f)

        #===============================================================
        #============================= DQN =============================
        #===============================================================
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
        Market.run_newEpisode(label[i-1,:].reshape(1, -1))  # test real FX rate
        state = Market.get_initial()  # get the initial state
        Market.start_transactions()  # start the transaction

        for t in range(T + 1):
            # select an action
            action = DQN_Agent.act(state, ifTest=True)
            # Market.action_list.append(action)
            next_state, reward, done, _ = Market.step(action, state)
            # next_state, reward, PL, done, _ = Market.step(action, state)
            Market.score += reward
            # Market.PL_score += PL
            # next_state to state
            state = next_state
            # transaction finished
            if done.all():
                break
        test_Naive_rewards = Market.Naive_reward()

        # test_LiqOpt_DQN_list, test_LiqOpt_DQN_std = captured_percentage_index(X_test, v_0, Market.score)
        # test_LiqOpt_Naive_list = captured_percentage_index(X_test, v_0, test_Naive_rewards)
        test_LiqOpt_DQN_list, DQN_std = captured_percentage_index(label[i-1,:].reshape(1, -1), v_0, Market.score)
        # test_LiqOpt_Naive_list = captured_percentage_index(np.expand_dims(label[i - 1], 0), v_0, test_Naive_rewards)
        CPI_DQN_list.append(test_LiqOpt_DQN_list)

        # # # estimation
        # Esti_PL, Esti_PL_mean, Esti_PL_std, Esti_PL_median, Esti_v_traj = stat_Esti_SDP(X_test, T, x_thres, v_min, v_max, v_0)
        # # SDP
        # SDP_PL, SDP_PL_mean, SDP_PL_std, SDP_PL_median, SDP_v_traj = stat_SDP(X_test, T, A_STAR, x_min, x_max, N_x, v_0, Delta_v)

        # estimation
        Esti_PL, Esti_PL_mean, Esti_PL_std, Esti_PL_median, Esti_v_traj = stat_Esti_SDP(label[i-1,:].reshape(1, -1), T, x_thres, v_min,
                                                                                        v_max, v_0)
        # SDP
        SDP_PL, SDP_PL_mean, SDP_PL_std, SDP_PL_median, SDP_v_traj = stat_SDP(label[i-1,:].reshape(1, -1), T, A_STAR, x_min, x_max, N_x, v_0, Delta_v)

        # Naive
        n = v_0 // v_min
        remain = v_0 - n * v_min
        interval = T // n
        # Naive_PL, Naive_PL_mean, Naive_PL_std, Naive_PL_median, Naive_v_traj = stat_Naive(X_test, v_min, T, interval, remain, v_0)
        Naive_PL, Naive_PL_mean, Naive_PL_std, Naive_PL_median, Naive_v_traj = stat_Naive(label[i-1,:].reshape(1, -1), v_min, T, interval, remain, v_0)

        # SDP_del_pl_mean, SDP_del_pl_std, SDP_del_pl_median, SDP_p_pl, SDP_glr = SDP_metrics(Naive_PL, SDP_PL)
        # Esti_del_pl_mean, Esti_del_pl_std, Esti_del_pl_median, Esti_p_pl, Esti_glr = SDP_metrics(Naive_PL, Esti_PL)

        # CPI_Esti_mean, CPI_Esti_std = captured_percentage_index(X_test, v_0, Esti_PL)
        # CPI_SDP_mean, CPI_SDP_std = captured_percentage_index(X_test, v_0, SDP_PL)
        # CPI_Naive_mean, CPI_Naive_std = captured_percentage_index(X_test, v_0, Naive_PL)
        CPI_Esti_mean, CPI_Esti_std = captured_percentage_index(label[i-1,:].reshape(1, -1), v_0, Esti_PL)
        CPI_SDP_mean, CPI_SDP_std = captured_percentage_index(label[i-1,:].reshape(1, -1), v_0, SDP_PL)
        CPI_Naive_mean, CPI_Naive_std = captured_percentage_index(label[i-1,:].reshape(1, -1), v_0, Naive_PL)
        CPI_Esti_list.append(CPI_Esti_mean)
        CPI_SDP_list.append(CPI_SDP_mean)
        CPI_Naive_list.append(CPI_Naive_mean)
    print('||DQN|| -- Captured percentage index: mean{:.4f}, std{:4f}\n'.format(np.mean(CPI_DQN_list), np.std(CPI_DQN_list)))
    print('||Esti-SDP|| -- Captured percentage index: mean{:.4f}, std{:4f}\n'.format(np.mean(CPI_Esti_list), np.std(CPI_Esti_list)))
    print('||SDP|| -- Captured percentage index: mean{:.4f}, std{:4f}\n'.format(np.mean(CPI_SDP_list), np.std(CPI_SDP_list)))
    print('||Naive|| -- Captured percentage index: mean{:.4f}, std{:4f}\n'.format(np.mean(CPI_Naive_list), np.std(CPI_Naive_list)))
    # print(
    #     '||SDP|| -- Expected rewards:  mean:{:.4f}, std:{:.4f}, median:{:.4f} \n        -- Delta_PL. mean:{:.4f}, std:{:.4f}, median:{:.4f}, P(Delta_PL > 0):{:.3f}, GLR:{:.4f}\n'.format(
    #         SDP_f_mean, SDP_f_std, SDP_f_median, SDP_PL_mean, SDP_PL_std, SDP_PL_median, SDP_P_PL, SDP_GLR))
    # print(
    #     '||Esti-SDP|| -- Expected rewards:  mean:{:.4f}, std:{:.4f}, median:{:.4f} \n             -- Delta_PL. mean:{:.4f}, std:{:.4f}, median:{:.4f}, P(Delta_PL > 0):{:.3f}, GLR:{:.4f}\n'.format(
    #         Esti_f_mean, Esti_f_std, Esti_f_median, Esti_PL_mean, Esti_PL_std, Esti_PL_median, Esti_P_PL, Esti_GLR))
    # print(
    #     '||Naive|| -- Expected rewards: mean:{:.4f}, std:{:.4f}, median:{:.4f}\n'.format(Naive_f_mean, Naive_f_std,
    #                                                                                          Naive_f_median))

####################################################################################
#                             Test with specific A, N, Sigma
####################################################################################
indi_test = False
# indi_test = True
if indi_test:
    currency = 'USD'
    expand_steps_ForTrain = 200  # the steps you wanna decoder to expand
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

    trend = 'up'
    # when i == 0 in calibration
    # # Downtrend
    if trend == 'down':
        # A = -0.02038755
        # N = -0.00205892
        # SIGMA = 0.0074969
        # START_PRICE = 0.030141156911986808

        # Real
        i_ = 19
        A = A_list[i_]
        N = N_list[i_]
        SIGMA = Sigma_list[i_]
        START_PRICE = x_0_List[i_]

    # # when i == 46
    # # Flat trend
    elif trend == 'flat':
        # A = -0.07930878
        # N = 0.01341533
        # SIGMA = 0.0038368
        # START_PRICE = 0.1655991806495659

        # Real
        i_ = 2
        A = A_list[i_]
        N = N_list[i_]
        SIGMA = Sigma_list[i_]
        START_PRICE = x_0_List[i_]

    # when i == 20
    # Up trend
    elif trend == 'up':
        # A = -0.05549867
        # N = 0.02086743
        # SIGMA = 0.00464528
        # START_PRICE = 0.30814624558413484

        # Real
        i_ = 0
        A = A_list[i_]
        N = N_list[i_]
        SIGMA = Sigma_list[i_]
        START_PRICE = x_0_List[i_]

    # when i == 0 in calibration
    # # Downtrend
    # A = -0.02038755
    # N = -0.00205892
    # SIGMA = 0.0074969
    # START_PRICE = 0.030141156911986808

    # when i == 46
    # Sideways
    # A = -0.07930878
    # N = 0.01341533
    # SIGMA = 0.0038368
    # START_PRICE = 0.1655991806495659

    # # when i == 20
    # # Up trend
    # A = -0.05549867
    # N = 0.02086743
    # SIGMA = 0.00464528
    # START_PRICE = 0.30814624558413484

    X = MC_simulation(0, M, T, START_PRICE, A, N, SIGMA)
    X_train, X_test = X[:1000, :], X[1000:, :]
    x_min, x_max = np.min(X), np.max(X)
    Delta_x = (x_max - x_min) / N_x

    # x_min_train, x_max_train = np.min(X_train), np.max(X_train)
    # x_min_test, x_max_test = np.min(X_test), np.max(X_test)
    # visualize_curves(np.exp(X), np.exp(x_min), np.exp(x_max), T, A, N)

    # address1 = './data/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}.pkl'.format(
    #         M // 2, N_pqc, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE)# read estimated SDP
    # with open(address1, 'rb') as f:
    #     f_star, f_history, x_thres = pkl.load(f)
    address1 = './data/Esti-SDP/Real_USD/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}_{}th.pkl'.format(
        M // 2, N_pqc, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE, i_+2)  # read estimated SDP
    with open(address1, 'rb') as f:
        f_star, f_history, x_thres = pkl.load(f)
    # read SDP
    address2 = './data/SDP/Real_USD/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_{}th.pkl'.format(
            N_x, N_v, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, i_ + 2)
    with open(address2, 'rb') as f:
        F_STAR, A_STAR = pkl.load(f)

    # estimation
    Esti_PL, Esti_PL_mean, Esti_PL_std, Esti_PL_median, Esti_v_traj = stat_Esti_SDP(X_test, T, x_thres, v_min,
                                                                                    v_max, v_0)
    # SDP
    SDP_PL, SDP_PL_mean, SDP_PL_std, SDP_PL_median, SDP_v_traj = stat_SDP(X_test, T, A_STAR, x_min, x_max, N_x, v_0,
                                                                          Delta_v)

    # Naive
    n = v_0 // v_min
    remain = v_0 - n * v_min
    interval = T // n
    Naive_PL, Naive_PL_mean, Naive_PL_std, Naive_PL_median, Naive_v_traj = stat_Naive(X_test, v_min, T, interval,
                                                                                          remain, v_0)

    # SDP_PL_mean, SDP_PL_std, SDP_PL_median, SDP_P_PL, SDP_GLR = SDP_metrics(Naive_f, SDP_f)
    # Esti_PL_mean, Esti_PL_std, Esti_PL_median, Esti_P_PL, Esti_GLR = SDP_metrics(Naive_f, Esti_f)

    CPI_Esti_mean, CPI_Esti_std = captured_percentage_index(X_test, v_0, Esti_PL)
    CPI_SDP_mean, CPI_SDP_std = captured_percentage_index(X_test, v_0, SDP_PL)
    CPI_Naive_mean, CPI_Naive_std = captured_percentage_index(X_test, v_0, Naive_PL)

    print('||Esti-SDP|| -- Captured percentage index: mean{:.4f}, std{:4f}\n'.format(CPI_Esti_mean,
                                                                                     CPI_Esti_std))
    print('||SDP|| -- Captured percentage index: mean{:.4f}, std{:4f}\n'.format(CPI_SDP_mean,
                                                                                CPI_SDP_std))
    print('||Naive|| -- Captured percentage index: mean{:.4f}, std{:4f}\n'.format(CPI_Naive_mean,
                                                                                  CPI_Naive_std))

    # print('||SDP|| -- Expected rewards:  mean:{:.4f}, std:{:.4f}, median:{:.4f} \n        -- Delta_PL. mean:{:.4f}, std:{:.4f}, median:{:.4f}, P(Delta_PL > 0):{:.3f}, GLR:{:.4f}\n'.format(SDP_f_mean, SDP_f_std, SDP_f_median, SDP_PL_mean, SDP_PL_std, SDP_PL_median, SDP_P_PL, SDP_GLR))
    # print('||Esti-SDP|| -- Expected rewards:  mean:{:.4f}, std:{:.4f}, median:{:.4f} \n             -- Delta_PL. mean:{:.4f}, std:{:.4f}, median:{:.4f}, P(Delta_PL > 0):{:.3f}, GLR:{:.4f}\n'.format(Esti_f_mean, Esti_f_std, Esti_f_median, Esti_PL_mean, Esti_PL_std, Esti_PL_median, Esti_P_PL, Esti_GLR))
    # print('||Naive|| -- Expected rewards: mean:{:.4f}, std:{:.4f}, median:{:.4f}\n'.format(Naive_f_mean, Naive_f_std, Naive_f_median))

    plot = False
    # plot = True
    if plot:
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        t1 = np.linspace(0, T, T + 1)
        t2 = np.linspace(0, T+1, T+2)
        ax1 = plt.subplot(231)
        ax1.set_xlabel('t')
        ax1.set_ylabel('v')
        ax1.plot(t2, Esti_v_traj[0], label='Esti-SDP')
        plt.legend()
        ax2 = plt.subplot(232)
        ax2.plot(t2, SDP_v_traj[0], label='SDP')
        plt.legend()
        ax3 = plt.subplot(233)
        ax3.plot(t2, Naive_v_traj[0], label='TWAP')
        plt.legend()
        ax3 = plt.subplot(234)
        ax3.set_xlabel('t')
        ax3.set_ylabel('x')
        ax3.plot(t1, X[0, :], label='X[0]')
        plt.legend()
        plt.show()




