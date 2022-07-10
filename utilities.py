import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time

###############       Calibrate A, N, Sigma        #####################
def calibration_fx(data, dt, if_log=True):
    """
    calibrate the a and b parameters of equation:
    y = a*x_{1} + b*x_{2} + c + \epsilon
    data: The Dataframe
    x2: x2 is the second feature
    if_log: If the data must be non-negative: True, else False
    :return: alpha, m, epsilon
    """

    # x1 is t
    # data_len = len(data['EURUSD_Curncy'])
    # time_ = np.arange(1.0, data_len).reshape(-1, 1) # the first feature is time

    y_t = np.array([])

    for i, key_ in enumerate(data):
        if if_log:
            data_i_ = np.log(np.array(np.copy(data[key_]))).reshape((-1, 1))
        else:
            data_i_ = np.array(np.copy(data[key_])).reshape((-1, 1))
        y_t = np.concatenate((y_t, data_i_), axis=1) if y_t.size else data_i_ #concatenate

    # y for LinearRegression
    label_ = np.array(y_t[1:, :] - y_t[:-1, :])
    # label_ = np.concatenate((time_, label_), axis=1)
    # x for LinearRegression
    # train_ = y_t[:-1, :].reshape(-1,1)
    train_ = y_t[:-1, :]
    # train_ = np.concatenate((time_, train_), axis=1)
    # if calibration_target in if_Non_Nega and if_Non_Nega[calibration_target]:
    #     x = np.log(np.array(np.copy(data[calibration_target][:-1]))).reshape(-1,1)
    # else:
    #     x = np.array(np.copy(data[calibration_target][:-1])).reshape(-1, 1)

    # LinearRegression fit
    reg = LinearRegression().fit(train_, label_)
    # predict
    y_pred = reg.predict(train_) # predict y using x
    epsilon = (label_ - y_pred)
    # calculate R2 score
    # R2_score = reg.score(train_, label_)

    # X2 = sm.add_constant(train_)
    # est = sm.OLS(label_[:,0], X2)
    # est2 = est.fit()
    # print(est2.summary())
    # print(est2.pvalues)

    # y = ax + b + epsilon
    # M = reg.coef_[:, 1] / dt
    # A = reg.coef_[:, 1:] / dt

    A = reg.coef_ / dt
    N = reg.intercept_ / dt

    if epsilon.shape[-1] == 1:
        L = np.std(epsilon)
    else:
        # Covariance Matrix of Epsilon
        cov_matrix = np.cov(np.transpose(epsilon))
        L = cholesky(cov_matrix / dt)

    # return M, A, N, L, R2_score
    return np.squeeze(A), np.squeeze(N), np.squeeze(L)

###############       Esti-SDP        #####################
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

def visualize_curves(X, x_min, x_max, T, A, N):
    mean_level = np.exp(N / (-A))
    plt.figure()
    t = np.linspace(0, T, T + 1)
    for m in range(X.shape[0]):
        plt.plot(t, X[m])
    plt.plot(t, [mean_level] * (T + 1), label='mean level')
    plt.plot(t, [x_min] * (T + 1), c='k')
    plt.plot(t, [x_max] * (T + 1), c='k')
    plt.plot(t, [(x_min + x_max) / 2] * (T + 1), label='middle of min, max')
    plt.legend()
    plt.show()


###############       DQN        #####################
def DQN_metrics(Epoch_Improve):
    """
    Calculate the metrics mean, std, median, GLR of \Delta P&L, P(\Delta P&L > 0)
    :param Epoch_Improve: The P&L over the dataset
    :return:
    """
    mean, median, std \
        = np.mean(Epoch_Improve), np.median(Epoch_Improve), np.std(Epoch_Improve)
    GLR = np.mean([n for n in Epoch_Improve if n > 0]) / \
          np.mean([-n for n in Epoch_Improve if n <= 0] + [1e-8])
    P_PL = len([n for n in Epoch_Improve if n > 0]) / len(Epoch_Improve)
    return mean, median, std, GLR, P_PL

def SDP_metrics(Naive, SDP):
    """
    calculate metrics of SDP, Esti-SDP
    :param Naive: expected rewards of naive strategy over samples
    :param SDP: expected rewards of SDP or Esti-SDP strategy over samples
    :return:
    """
    M = SDP.shape[0]
    Delta_PL = ((SDP - Naive) / Naive) * 100
    mean, std, median = np.mean(Delta_PL), np.std(Delta_PL), np.median(Delta_PL)
    P_PL = len([Delta_PL[i] for i in range(M) if Delta_PL[i] > 0]) / M
    GLR = np.mean([Delta_PL[i] for i in range(M) if Delta_PL[i] > 0]) / \
          np.mean([-Delta_PL[i] for i in range(M) if Delta_PL[i] <= 0] + [1e-8])
    return mean, std, median, P_PL, GLR

def stat_Esti_SDP(X, T, x_thres, v_min, v_max, v_0):
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
    for m in range(X.shape[0]):
        v = v_0
        for t in range(T + 1):
            a = estimated_a_star(x_thres, X[m, t], v, t, T, v_min, v_max)
            f[m] += a / np.exp(X[m, t])
            v = v - a
            v_traj[m, t + 1] = v
    return f, np.mean(f), np.std(f), np.median(f), v_traj

def stat_SDP(X, T, A_STAR, x_min, x_max, N_x, v_0, Delta_v):
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
    for m in range(X.shape[0]):
        v = v_0
        for t in range(T + 1):
            if int((X[m, t]-x_min)//Delta_x) < 0:
                a = A_STAR[0, int(v//Delta_v), t]
            else:
                a = A_STAR[min(int((X[m, t]-x_min)//Delta_x), A_STAR.shape[0]-1), int(v//Delta_v), t]
            f[m] += a / np.exp(X[m, t])
            v = v - a
            v_traj[m, t + 1] = v
    return f, np.mean(f), np.std(f), np.median(f), v_traj

def stat_Naive(X, v_min, T, interval, remain, v_0):
    f = np.zeros([X.shape[0]], dtype=np.float32)
    v_traj = np.zeros([X.shape[0], T + 2], dtype=np.float32)
    v_traj[:, 0] = v_0
    # start = time.time()
    for m in range(X.shape[0]):
        v = v_0
        cnt = 0
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
        # sell remain at T
        f[m] += remain / np.exp(X[m, T])
    # end = time.time()
    # print('TWAP time cost: {}'.format(end - start))
    return f, np.mean(f), np.std(f), np.median(f), v_traj

def captured_percentage_index(X, v_0, total_rewards):
    # FX_capture = log(v0 / total reward)
    # 100 - [(FX_capture - FX_min) / (FX_max - FX_min) * 100]
    n = len(total_rewards)
    CPI = []
    for i in range(n):
        x_min, x_max = np.min(X[i]), np.max(X[i])
        FX_capture = (v_0 / np.squeeze(total_rewards[i]))
        CPI.append(1 - ((FX_capture - np.exp(x_min)) / (np.exp(x_max) - np.exp(x_min))))
    return np.mean(CPI), np.std(CPI)