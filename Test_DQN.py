from Environment import *
from DQN_Agent import *
import pandas as pd
import numpy as np
import pickle as pkl
import os.path
import torch
import os
import time
from utilities import *

import 	torch
import  time


def trading(days, action_list, reward_list, score, cur_state):
    for i in range(days):
        # if i == 70:
        #     print(1)
        action = DQN_Agent.act(cur_state)
        action_list.append(action)

        next_state, reward, done, info = Market.step(action, cur_state)
        reward_list.append(reward)

        score += reward

        # if done, next state is None
        if not done:
            next_state = None

        DQN_Agent.step(cur_state, action, reward, next_state, done)

        cur_state = next_state

        if done:
            break
    return score

# ----------------------------- Parameters for DQN ----------------------------- #
STATE_SIZE = 3   # (x, v, t)
ACTION_SIZE = 17 # [0, V_MIN, V_MIN + 1, ..., V_MAX]
FC1_UNITS, FC2_UNITS, FC3_UNITS, FC4_UNITS, FC5_UNITS = [64] * 5
LR = 0.00025
MOMENTUM = 0.95
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32
CURVE_BATCH_SIZE = 10
UPDATE_EVERY = 10000
TAU = 0.1              # for soft update of target network
EPS_START = 0.95         # start value for epsilon-greedy decay
EPS_END = 0.          # end value for epsilon-greedy decay
EPS_DECAY = 50000         # decay every
RANDOM_SEED = 0
# Early_stop = 20

V_0 = 100
N_V = 100
N_X = 100
DELTA_V = V_0 / N_V
V_MIN = 10
V_MAX = 25
T = 100
M = 2000
Epochs = 800
# when i == 0 in calibration
# # Downtrend
# A = -0.02038755
# N = -0.00205892
# SIGMA = 0.0074969
# START_PRICE = 0.030141156911986808

# when i == 46
# Flat trend
# A = -0.07930878
# N = 0.01341533
# SIGMA = 0.0038368
# START_PRICE = 0.1655991806495659

# when i == 20
# Up trend
# A = -0.05549867
# N = 0.02086743
# SIGMA = 0.00464528
# START_PRICE = 0.30814624558413484
# ----------------------------- Generate simulated FX data ----------------------------- #
# with open('./data/DQN/Best_Optimality_{}.pkl'.format(FC1_UNITS), 'rb') as f:
#     Max_TestOptimality = pkl.load(f)

currency= 'CNY'
StartIndex = 0
Interval = 100
len_= 250
train = 1

# with open('./data/DQN/Optimality_Curve_{}_units{}_update{}.pkl'.format(currency, FC1_UNITS, UPDATE_EVERY), 'rb') as f:
#     curves = pkl.load(f)
# plt.figure()
# t = np.linspace(0, len(curves[0]), len(curves[0]))
# plt.plot(t, curves[0], label='training')
# plt.plot(t, curves[1], label='test')
# plt.legend()
# plt.show()

# real FX data
expand_steps_ForTrain = 200 # the steps you wanna decoder to expand
A_list, N_list, Sigma_list, x_0_List, idx_list = [], [], [], [], []
for i in range(1, 96):
    with open('./experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_AvgANSigma.pkl'.format(
            expand_steps_ForTrain, currency, i, currency), 'rb') as f:
        A, N, SIGMA, START_PRICE, LABEL, PRED_RESULTS = pkl.load(f)
    idx_list.append(i)
    A_list.append(A)
    N_list.append(N)
    Sigma_list.append(SIGMA)
    x_0_List.append(START_PRICE)

if train:
    # with open("./data/ANSigma/{}_ANSigmaX0_Intval{}_StartIndex{}_len{}.pkl".format(currency, Interval, StartIndex, len_), "rb") as f:
    #     A_list, N_list, Sigma_list, x_0_List = pkl.load(f)
    for i, A, N, SIGMA, START_PRICE in zip(idx_list, A_list, N_list, Sigma_list, x_0_List):
        if i != 12:
            continue
    # for i in range(0, len(A_list)):
    # for i in range(0, 1):
    #     A, N, SIGMA, START_PRICE = A_list[i], N_list[i], Sigma_list[i], x_0_List[i]

        print("---------- {}th Begin ------------\n".format(i))
        Max_TestOptimality = -float('inf')
        X = MC_simulation(0, M, T, START_PRICE, A, N, SIGMA)
        X_train, X_test = X[:1000, :], X[1000:, :]
        # x_min, x_max = np.min(X), np.max(X)

# if os.path.isfile("./data/DQN/Best_HyperParams_{}.pkl".format(currency)):
#     with open("./data/DQN/Best_HyperParams_{}.pkl".format(currency), "rb") as f:
#         saved_Best_Hyper = pkl.load(f)
#     overall_best_optimality = saved_Best_Hyper[-1]
# else:
#     overall_best_optimality = -float('inf')
# Best_Hyper = []
# print('units{}, LR{}, TAU{}, EPS_DECAY{} Start'.format(units, LR, TAU, EPS_DECAY))

        # ----------------------------- Market Env ----------------------------- #
        Market = MarketEnvironment(random = RANDOM_SEED,
                                  liquid_time = T,
                                  total_vol = V_0,
                                  v_min = V_MIN,
                                  v_max = V_MAX)
        # ----------------------------- DQN Agent ----------------------------- #
        DQN_Agent = Agent(STATE_SIZE, ACTION_SIZE,
                          FC1_UNITS, FC2_UNITS, FC3_UNITS, FC4_UNITS, FC5_UNITS,
                          LR, MOMENTUM, BUFFER_SIZE, BATCH_SIZE, UPDATE_EVERY, TAU,
                          EPS_START, EPS_END, EPS_DECAY, RANDOM_SEED)

        # load pretrained model, if exist
        saved_model_path = './model/DQN/Real_{}/DQN_Model_{}_State{}Action{}_LR{}_5FC{}_EPS_DECAY{}_Update{}_Tau{}_{}th.pkl'.format(
            currency, currency, STATE_SIZE, ACTION_SIZE, LR, FC1_UNITS, EPS_DECAY, UPDATE_EVERY, TAU, i)

        checkpoint = torch.load(saved_model_path)
        DQN_Agent.qnetwork_policy.load_state_dict(checkpoint['DQN_Policy_state_dict'])
        DQN_Agent.qnetwork_target.load_state_dict(checkpoint['DQN_Target_state_dict'])
        DQN_Agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        DQN_Agent.steps_done = 0 # step for updating eps_threshold
        # for e in range(NUM_EPOCHS):
        e = 0
        ES_cnt = 0 # early stop
        TrainOptimality_Curve, TestOptimality_Curve = [], []
        while True:
            # shuffle
            start = time.time()
            np.random.shuffle(X_train)
            # Epoch_Improve_Trian = []
            # Sum_Rewards_Train = []
            PL_DQN_list, PL_Naive_list = [], []
            for b in range(0, X_train.shape[0], CURVE_BATCH_SIZE):
                # reset
                Market.reset(random=RANDOM_SEED,
                             liquid_time=T,
                             total_vol=V_0,
                             v_min=V_MIN,
                             v_max=V_MAX)
                # Initilization
                Market.run_newEpisode(X_train[b:b + CURVE_BATCH_SIZE])  # run one episode
                state = Market.get_initial()  # get the initial state
                Market.start_transactions()  # start the transaction

                for t in range(T + 1):
                    # select an action
                    # if t == T-1:
                    #     a = 1
                    action = DQN_Agent.act(state, ifTest=False)
                    Market.action_list.append(action)
                    # Profit & Loss PL = volume * price
                    # next_state, reward, PL, done, _ = Market.step(action, state)
                    next_state, reward, done, _ = Market.step(action, state)
                    Market.reward_list.append(reward)
                    Market.score += reward
                    # Market.PL_score += PL
                    # judge the next_state
                    # if done:
                    #     next_state = None
                    # Save the transitions
                    DQN_Agent.step(state, action, reward, next_state)
                    # next_state to state
                    state = next_state
                    # transaction finished
                    if done.all():
                        break
                # calculate the rewards of Naive policy
                Naive_rewards = Market.Naive_reward()
                PL_DQN_list.extend(Market.score.tolist())
                PL_Naive_list.extend(Naive_rewards.tolist())

            LiqOpt_DQN_list = captured_percentage_index(X_train, V_0, PL_DQN_list)
            LiqOpt_Naive_list = captured_percentage_index(X_train, V_0, PL_Naive_list)
            TrainOptimality_Curve.append(np.mean(LiqOpt_DQN_list))
            print('Epoch {}\n'.format(e))
            print('||DQN|| -- Avg. total rewards: mean{:.4f}, std{:.4f}\n'.format(np.mean(PL_DQN_list), np.std(PL_DQN_list)))
            print('        -- liquidation optimality: mean{:.4f}, std{:.4f}\n'.format(np.mean(LiqOpt_DQN_list), np.std(LiqOpt_DQN_list)))
            print('||Naive|| -- Avg. total rewards: mean{:.4f}, std{:.4f}\n'.format(np.mean(PL_Naive_list), np.std(PL_Naive_list)))
            print('          -- liquidation optimality: mean{:.4f}, std{:.4f}\n'.format(np.mean(LiqOpt_Naive_list), np.std(LiqOpt_Naive_list)))

            # print('Scores: {} \n\n'.format(Market.reward_list))
            e += 1

            # ----------------------------- Testing ----------------------------- #

            # test_CPI_DQN_list, test_CPI_Naive_list = [], []
            # reset
            Market.reset(random=RANDOM_SEED,
                         liquid_time=T,
                         total_vol=V_0,
                         v_min=V_MIN,
                         v_max=V_MAX)
            # Initilization
            Market.run_newEpisode(X_test)  # run one episode
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

            test_LiqOpt_DQN_list = captured_percentage_index(X_test, V_0, Market.score)
            test_LiqOpt_Naive_list = captured_percentage_index(X_test, V_0, test_Naive_rewards)
            TestOptimality_Curve.append(np.mean(test_LiqOpt_DQN_list))

            end = time.time()

            print('------------------- Test ------------------\n')
            print('||DQN|| -- Test Avg. total rewards: {:.4f}, std{:.4f}\n'.format(np.mean(Market.score), np.std(Market.score)))
            print('        -- Test liquidation optimality: mean{:.4f}, std{:.4f}\n'.format(np.mean(test_LiqOpt_DQN_list), np.std(test_LiqOpt_DQN_list)))
            print('||Naive|| -- Test Avg. total rewards: {:.4f}, std{:.4f}\n'.format(np.mean(test_Naive_rewards), np.std(test_Naive_rewards)))
            print('          -- Test liquidation optimality: mean{:.4f}, std{:.4f}\n'.format(np.mean(test_LiqOpt_Naive_list), np.std(test_LiqOpt_Naive_list)))
            print('time cost for 1 epoch: {}\n'.format(end - start))

            if np.mean(test_LiqOpt_DQN_list) > Max_TestOptimality:
                Max_TestOptimality = np.mean(test_LiqOpt_DQN_list)
                # ----------------------------- Save Model ---- 22 ------------------------- #
                # model_path = './model/DQN/{}/DQN_Model_{}_State{}Action{}_LR{}_5FC{}_EPS_DECAY{}_Update{}_Tau{}_{}th.pkl'.format(
                #     currency, currency, STATE_SIZE, ACTION_SIZE, LR, FC1_UNITS, EPS_DECAY, UPDATE_EVERY, TAU, i)
                model_path = './model/DQN/Real_{}/DQN_Model_{}_State{}Action{}_LR{}_5FC{}_EPS_DECAY{}_Update{}_Tau{}_{}th.pkl'.format(
                    currency, currency, STATE_SIZE, ACTION_SIZE, LR, FC1_UNITS, EPS_DECAY, UPDATE_EVERY, TAU, i)
                torch.save({'DQN_Policy_state_dict': DQN_Agent.qnetwork_policy.state_dict(),
                            'DQN_Target_state_dict': DQN_Agent.qnetwork_target.state_dict(),
                            'optimizer_state_dict': DQN_Agent.optimizer.state_dict(),
                            }, model_path)
     
            if e + 1 % 50 == 0:
                with open('./data/DQN/Real_{}/Optimality_Curve_{}_units{}_update{}_{}th.pkl'.format(currency, currency, FC1_UNITS, UPDATE_EVERY, i),
                          'wb') as f:

                    pkl.dump([TrainOptimality_Curve, TestOptimality_Curve], f)
                # with open('./data/DQN/{}/Optimality_Curve_{}_units{}_update{}_{}th.pkl'.format(currency, currency,
                #                                                                                     FC1_UNITS,
                #                                                                                     UPDATE_EVERY, i),
                #           'wb') as f:
                #     pkl.dump([TrainOptimality_Curve, TestOptimality_Curve], f)
            if e >= Epochs:
                break

        print("---------- {}th Finished ------------\n\n".format(i))


        # if np.mean(test_LiqOpt_DQN_list) > Max_TestOptimality:
        #     Max_TestOptimality = np.mean(test_LiqOpt_DQN_list)
        #     ES_cnt = 0
        #     with open('./data/DQN/Best_Optimality_{}.pkl'.format(FC1_UNITS), 'wb') as f:
        #         pkl.dump(Max_TestOptimality, f)
        # else:
        #     ES_cnt += 1

        # if ES_cnt > Early_stop:
        #     print('Epochs{}, units{}, LR{}, TAU{}, EPS_DECAY{} Finished\n\n'.format(e, FC1_UNITS, LR, TAU, EPS_DECAY))
        #     break


        # print('Test / Epoch Avg. Improvement: {0:.3f}'.format(Mean_test))
        # print('Median: {0:.3f}'.format(Median_test))
        # print('std: {0:.3f}'.format(Std_test))
        # print('GLR: {0:.3f}'.format(GLR_test))
        # print('P_PL: {0:.3f} \n'.format(P_PL_test))
        # print('Expected total rewards: {}\n\n'.format(np.mean(Sum_Rewards_Record_Test)))
        # with open(exp_path, 'wb') as f:
            #     pickle.dump(DQN_Agent.memory.memory, f)
    # max_idx = np.argmax([each[0] for each in Improvement_Record_Train])
    # cross_valid.append([Improvement_Record_Train[max_idx], Improvement_Record_Test[max_idx]])
    # print(cross_valid)
    # result_path = './results/DQN_CrossVal_State{}Action{}_LR{}_5FC{}.pkl'.format(STATE_SIZE, ACTION_SIZE,
    #                                                                LR, FC1_UNITS)
    # with open(result_path, 'wb') as f:
    #     pickle.dump(cross_valid, f)

# result_path = './results/DQN/DQN_State{}Action{}_LR{}_5FC{}.pkl'.format(STATE_SIZE, ACTION_SIZE,
#                                                                LR, FC1_UNITS)
# with open(result_path, 'rb') as f:
#     cross_valid = pickle.load(f)


# # ----------------------------- Save ReplayMemory ----------------------------- #
# exp_path = './data/DQN_ReplayMemo_State{}Action{}_LR{}_5FC{}.pkl'.format(STATE_SIZE, ACTION_SIZE,
#                                                                LR, FC1_UNITS)
# with open(exp_path, 'wb') as f:
#     pickle.dump(DQN_Agent.memory.memory, f)
# # ----------------------------- Save Results ----------------------------- #
# result_path = './results/DQN_Results_State{}Action{}_LR{}_5FC{}.pkl'.format(STATE_SIZE, ACTION_SIZE,
#                                                                LR, FC1_UNITS)
# with open(result_path, 'wb') as f:
#     pickle.dump({'train': Improvement_Record_Train, 'test': Improvement_Record_Test}, f)
# # ----------------------------- Plot ----------------------------- #
# plot_result([[Improvement_Record_Train[i][0] for i in range(NUM_EPOCHS)],
#              [Improvement_Record_Test[i][0] for i in range(NUM_EPOCHS)]],
#             [[Improvement_Record_Train[i][1] for i in range(NUM_EPOCHS)],
#              [Improvement_Record_Test[i][1] for i in range(NUM_EPOCHS)]],
#             [[Improvement_Record_Train[i][2] for i in range(NUM_EPOCHS)],
#              [Improvement_Record_Test[i][2] for i in range(NUM_EPOCHS)]],
#             [[Improvement_Record_Train[i][3] for i in range(NUM_EPOCHS)],
#              [Improvement_Record_Test[i][3] for i in range(NUM_EPOCHS)]],
#             [[Improvement_Record_Train[i][4] for i in range(NUM_EPOCHS)],
#              [Improvement_Record_Test[i][4] for i in range(NUM_EPOCHS)]])



 





