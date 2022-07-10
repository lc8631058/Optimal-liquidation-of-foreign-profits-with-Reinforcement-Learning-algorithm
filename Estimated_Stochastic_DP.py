import matplotlib.pyplot as plt
import numpy as np
import time
import pickle as pkl
import matplotlib.animation as animation
from matplotlib import cm

def function(t, p, q, c):
    """
    p,q,c function
    :param p,q,c: parameters
    """
    # t is a vector
    return p * (t * t) + q * t + c

def pqc_condition(p, q, c):
    """
    judge if the parameters p,q,c are legal
    """
    if p < 0 or q < 0 or c < x_min or p * (T**2) + q * T + c > x_max:
        return False
    else:
        return True

def generate_pqc(N, T, x_min, x_max):
    """
    Generate N (p,q,c) pairs over time T given price limit (x_min, x_max)
    :param N: number of pairs needed to be generated
    :param T: time horizon
    :param x_min, x_max: price limit
    """
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

def optimal_strategy_estimation(X, R, v_min, v_max):
    """
    Estimate the optimal strategy given (p,q,c) pairs under volume limit (v_min, v_max)
    :param X: prices
    :param R: (p,q,c) pairs
    :param v_min, v_max: volume limit
    """
    M = X.shape[0]
    len_R = len(R)
    X_thres = np.zeros([len_R, T + 1], dtype=np.float32)
    x_thres = np.zeros([T + 1], dtype=np.float32)
    t = np.linspace(0, T, T + 1)
    for i in range(len_R):
        p, q, c = R[i]
        X_thres[i] = p * (t * t) + q * t + c
    f_star = -float('inf')
    f_history = np.zeros([len_R])
    time_cost = 0
    for i in range(len_R):
        start = time.time()
        f = np.zeros([M], dtype=np.float32)
        for m in range(M):
            v = v_0
            for t in range(T + 1):
                if t == T:
                    a = v
                else:
                    a = np.heaviside(X_thres[i, t] - X[m, t], 1) * \
                        (min(v_max, max(0, v - v_min) + v_min * np.heaviside(v - v_min, 1) * np.heaviside(v_max - v, 1)))
                f[m] += a / np.exp(X[m, t])
                v = v - a
        mean_f = np.mean(f)
        f_history[i] = mean_f
        if mean_f > f_star:
            f_star = mean_f
            x_thres = X_thres[i, :]
        end = time.time()
        time_cost += end - start
        print('{}/{} time cost: {:.2f}s'.format(i + 1, len_R, time_cost))
    return f_star,f_history, x_thres

def estimated_a_star(x_thres, x, v, t):
    """
    return estimated a*
    :param x_thres: estimated optimal threshold of x over t
    :param x: price
    :param v: volume
    :param t: time
    """
    if t == T:
        return v
    else:
        return np.heaviside(x_thres[t] - x, 1) * \
               (min(v_max, max(0, v - v_min) + v_min * np.heaviside(v - v_min, 1) * np.heaviside(v_max - v, 1)))

def MC_simulation(seed, M, T, START_PRICE, A, N, SIGMA):
    """
    Simulate trajectories by parameters A,N,Sigma
    :param seed: random seed
    :param M: number of trajectories
    :param T: time horizon
    :param START_PRICE: as the name
    :A, N, Sigma: parameters from OU process
    """
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

def visualize_curves(X, T):
    plt.figure()
    t = np.linspace(0, T, T + 1)
    for m in range(X.shape[0]):
        plt.plot(t, X[m])
    plt.show()

v_0 = 100
N_v = 100
N_x = 100
N_pqc = 10
Delta_v = v_0 / N_v
v_min = 10
v_max = 25

T = 100
M = 2000

# when i == 0 in calibration
# # Downtrend
# A = -0.02038755
# N = -0.00205892
# SIGMA = 0.0074969
# START_PRICE = 0.030141156911986808

# # when i == 46
# # Flat trend
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

currency= 'USD'
StartIndex = 0
Interval = 100
len_= 250
# with open("./data/ANSigma/{}_ANSigmaX0_Intval{}_StartIndex{}_len{}.pkl".format(currency, Interval, StartIndex, len_), "rb") as f:
#     A_list, N_list, Sigma_list, x_0_List = pkl.load(f)

# Train with real data
expand_steps_ForTrain = 200 # the steps you wanna decoder to expand
A_list, N_list, Sigma_list, x_0_List, idx_list = [], [], [], [], []
for i in range(1, 96):
    with open('./experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_AvgANSigma.pkl'.format(
            expand_steps_ForTrain, currency, i, currency), 'rb') as f:
        A, N, SIGMA, START_PRICE = pkl.load(f)
    idx_list.append(i)
    A_list.append(A)
    N_list.append(N)
    Sigma_list.append(SIGMA)
    x_0_List.append(START_PRICE)


# Esti = True
Esti = False
if Esti:
    for i, A, N, SIGMA, START_PRICE in zip(idx_list, A_list, N_list, Sigma_list, x_0_List):
        X = MC_simulation(0, M, T, START_PRICE, A, N, SIGMA)
        X_train, X_test = X[:1000, :], X[1000:, :]
        x_min, x_max = np.min(X), np.max(X)
        Delta_x = (x_max - x_min) / N_x
        # visualize_curves(X_test, T)

        R = generate_pqc(N_pqc, T, x_min, x_max)
        f_star, f_history, x_thres = optimal_strategy_estimation(X_train, R, v_min, v_max)
        with open('./data/Esti-SDP/Real_{}/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}_{}th.pkl'.format(
                currency, M // 2, N_pqc, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE, i), 'wb') as f:
            pkl.dump([f_star, f_history, x_thres], f)
        print('{}th finished\n\n'.format(i))

azim = 355
draw = False
# draw = True
if draw:
    with open('./data/x_thres_M{}_Npqc{}_T{}_v0{}_vmin{}_vmax{}_xmin{:.5f}_xmax{:.5f}_ANSIGMA{:.5f}-{:.5f}-{:.5f}_price{:.5f}.pkl'.format(M // 2, N_pqc, N_x, N_v, T, v_0, v_min, v_max, x_min, x_max, A, N, SIGMA, START_PRICE), 'rb') as f:
        f_star, f_history, x_thres = pkl.load(f)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.linspace(x_min, x_min + 101 * Delta_x, 101)
    ax.set_xlabel('$x$', fontsize=15)
    Y = np.arange(0, 201 * Delta_v, Delta_v)
    ax.set_ylabel('$v$', fontsize=15)
    X, Y = np.meshgrid(X, Y)
    ax.set_zlabel('$a*$', fontsize=15, rotation=0)

    ax.view_init(elev=10., azim=azim)
    imgs = []
    for t in range(T + 1):
        a_star = np.zeros([N_x + 1, N_v + 1], dtype=np.float32)
        # Plot the surface.
        for i in range(N_x + 1):
            x = x_min + i * Delta_x
            for j in range(N_v + 1):
                v = j * Delta_v
                a_star[i, j] = estimated_a_star(x_thres, x, v, t)
        # a = estimated_a_star(x_thres, x, v, t)
        surf = ax.plot_surface(X, Y, a_star.T, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # plt.savefig('./3d_plots/Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_{}_exp.jpg'.format(N_x, N_v, T, v_0, v_min, v_max,t))
        imgs.append([surf])
    ani = animation.ArtistAnimation(fig, imgs)
    writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('./3d_plots/Exp_optimal_actions_Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}_azim{}_ANSIGMA{:.3f}-{:.3f}-{:.3f}_estimated.mp4'
             .format(N_x, N_v, T, v_0, v_min, v_max, azim, A, N, SIGMA), writer=writer)

