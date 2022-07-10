import numpy as np
from scipy.stats import norm
import pickle as pkl
import time
from utilities import *
from decimal import Decimal
import pickle

def getP_i_prime(i_prime, mu, sigma):
    # Calculate P(x_i'|x_i) given i'
    if i_prime == 0:
        P_i_prime = norm(0, 1).cdf((x_min - mu) / sigma)
    elif 1 <= i_prime <= N_x - 1:
        P_i_prime = norm(0, 1).cdf((x_min + i_prime * Delta_x - mu) / sigma) - \
                    norm(0, 1).cdf((x_min + (i_prime - 1) * Delta_x - mu) / sigma)
    else:
        P_i_prime = 1 - norm(0, 1).cdf((x_max - Delta_x - mu) / sigma)
    return P_i_prime

def stochastic_DP(x_min, N_x, Delta_x, v_min, v_max, N_v, Delta_v, T, A, N, Sigma):
    """
    Stochastic Dynamic Programming
    :param x_min: x_min
    :param N_x: Number of steps for price x
    :param Delta_: \Delta x = (x_max - x_min) / N_x
    :param v_min: if sell, minimum volume to sell
    :param v_max: if sell, maximum volume to sell
    :param N_v: Number of steps for discretization
    :param Delta_v: \Delta v = v_0 / N_v
    :param T: Total time steps
    :param A: param of GOU
    :param N: param of GOU
    :param Sigma: param of GOU 
    :return: optimal policy a*
    """
    F_star = np.zeros([N_x + 1, N_v + 1, 2], dtype=np.float32)
    A_star = np.zeros([N_x + 1, N_v + 1, T + 1], dtype=np.int32)
    # s_i,j,t = [x_i, v_j, t]
    x_i = np.array([x_min + i * Delta_x for i in range(N_x + 1)])
    mu = np.array([(A + 1) * xi + N for xi in x_i])
    sigma = Sigma
    # For i, i', calculate all possible P(x_i'|x_i)
    P_i_prime_mat = np.zeros([N_x + 1, N_x + 1], dtype=np.float32)
    for i in range(N_x + 1):
        for i_prime in range(N_x + 1):
            P_i_prime_mat[i, i_prime] = getP_i_prime(i_prime, mu[i], sigma)

    for t in range(T, -1, -1):
        start = time.time()
        for i in range(N_x + 1):
            for j in range(N_v + 1):
                # A(s_i,j,t) action space
                # at end step, sell all j you hold
                if t == T:
                    a_ijt = j * Delta_v
                    r_ijt = a_ijt / np.exp(x_i[i])
                    F_star[i, j, 0] = r_ijt
                    A_star[i, j, t] = a_ijt
                # at 0,...,T-1, choose a j' from 0 to j as a_t
                else:
                    # best action and expected reward at s_i,j,t
                    a_star_ijt = None
                    f_star_ijt = -float('inf')
                    if j == 9:
                        d = 1
                    # action_diff = []

                    # action space of ijt
                    A_ijt = [0]
                    if int(v_min // Delta_v) <= j <= int(v_max // Delta_v):
                        A_ijt += list(range(int(v_min // Delta_v), int(j // Delta_v) + 1))
                    elif j > int(v_max // Delta_v):
                        A_ijt += list(range(int(v_min // Delta_v), int(v_max // Delta_v) + 1))
                    for a_j in A_ijt:

                        a_ijt = a_j * Delta_v
                        r_ijt = a_ijt / np.exp(x_i[i])

                        # For j', calculate P(v_j'|v_j, a_t)
                        P_j_prime = np.zeros([1, N_v + 1], dtype=np.float32)
                        P_j_prime[0, j - a_j] = 1.
                        # Sum_{i_prime, j_prime} P(x_i'|x_i) * P(v_j'|v_j, a_t) * f*_t+1(s_i',j',t+1)
                        weighted_f_ijt_next = (np.matmul(P_i_prime_mat[i, :].reshape([-1, 1]), P_j_prime)
                                               * F_star[:, :, 1]).sum()
                        # action_diff.append(r_ijt + weighted_f_ijt_next)
                        # update f*(s_ijt) to optimal
                        if r_ijt + weighted_f_ijt_next > f_star_ijt:
                            f_star_ijt = r_ijt + weighted_f_ijt_next
                            a_star_ijt = a_ijt
                    F_star[i, j, 0] = f_star_ijt
                    A_star[i, j, t] = a_star_ijt
        F_star[:, :, 1] = F_star[:, :, 0]
        end = time.time()
        print('t = {} processed\n'.format(t))
        print("time cost: {}\n".format(end - start))
    return F_star, A_star

# x = x_min + i * Delta_x, x is the log(FX rate)
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

# N_x = 400
# Delta_x = (x_max - x_min) / N_x

currency= 'USD'
StartIndex = 0
Interval = 100
len_= 250
# with open("./data/ANSigma/{}_ANSigmaX0_Intval{}_StartIndex{}_len{}.pkl".format(currency, Interval, StartIndex, len_), "rb") as f:
#     A_list, N_list, Sigma_list, x_0_List = pickle.load(f)

v_0 = 100
N_v = 100
Delta_v = v_0 / N_v
# index j, integer
v_min = 10
v_max = 25

T = 100
M = 2000

# Train with real data
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
# visualize_curves(np.exp(X), T)
# x_min = float(Decimal(x_min.item()).quantize(Decimal("0.001"), rounding = "ROUND_DOWN"))
# x_max = float(Decimal(x_max.item()).quantize(Decimal("0.001"), rounding = "ROUND_UP"))
# for ii in [19]:
#     i, A, N, SIGMA, START_PRICE = idx_list[ii], A_list[ii], N_list[ii], Sigma_list[ii], x_0_List[ii]
# for i, A, N, SIGMA, START_PRICE in zip(idx_list, A_list, N_list, Sigma_list, x_0_List):
for i, (A, N, SIGMA, START_PRICE) in enumerate(zip(A_list, N_list, Sigma_list, x_0_List)):
    # generate 2000 curves, 1000 for training, 1000 for testing
    X = MC_simulation(0, M, T, START_PRICE, A, N, SIGMA)
    X_train, X_test = X[:1000, :], X[1000:, :]
    x_min, x_max = np.min(X), np.max(X)
    # for N_x in [50, 200, 400, 600]:
    for N_x in [100]:
        Delta_x = (x_max - x_min) / N_x

        F_STAR, A_STAR = stochastic_DP(x_min, N_x, Delta_x, v_min, v_max, N_v, Delta_v, T, A, N, SIGMA)
        # # save
        address = './data/SDP/Real_{}/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_{}th.pkl'.format(
            currency, N_x, N_v, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, i)
        with open(address, 'wb') as f:
            pkl.dump([F_STAR, A_STAR], f)
    print('{}th finished\n\n'.format(i))

# load
# with open('./data/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}.pkl'.format(N_x, N_v, T, v_0, v_min, v_max), 'rb') as f:
#     F_STAR, A_STAR = pkl.load(f)


