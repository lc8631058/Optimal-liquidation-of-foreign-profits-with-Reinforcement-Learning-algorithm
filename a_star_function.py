from sklearn import tree
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import graphviz
import pydot
from sklearn.tree import export_text


# load
x_min = 0.77
x_max = 1.07
N_x = 100
Delta_x = (x_max - x_min) / N_x

v_0 = 100
N_v = 200
Delta_v = v_0 / N_v

T = 200
v_min = 20
v_max = 50
with open('./data/Exp_Nx{}_Nv{}_T{}_v0{}_vmin{}_vmax{}.pkl'.format(N_x, N_v, T, v_0, v_min, v_max), 'rb') as f:
    F_STAR, A_STAR = pkl.load(f)

def a_star_function(x_min, Delta_x, v_min, v_max, Delta_v):
    """
    a^* function
    :param v_min: min volume
    :param v_max: max volume
    :return: optimal action a*
    """
    x_thres = x_min + (1/2 * N_x + 1) * Delta_x
    a_star = np.zeros([N_x + 1, N_v + 1], dtype=np.float32)
    # price x
    for i in range(N_x + 1):
        x = x_min + i * Delta_x
        # volume v
        for j in range(N_v + 1):
            v = j * Delta_v
            #                     1   if x1 > 0
            # heaviside(x1, x2) = x2  if x1 == 0,
            #                     0   if x1 < 0
            a_star[i, j] = np.heaviside(x_thres - x, 1) * \
                     (min(v_max, max(0, v - v_min) + v_min * np.heaviside(v - v_min, 1) * np.heaviside(v_max - v, 1)))
    return a_star

a_star = a_star_function(x_min, Delta_x, int(v_min * Delta_v), int(v_max * Delta_v), Delta_v)

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(x_min, x_min + 101 * Delta_x, Delta_x)
ax.set_xlabel('$x$', fontsize=15)
Y = np.arange(0, 201 * Delta_v, Delta_v)
ax.set_ylabel('$v$', fontsize=15)
X, Y = np.meshgrid(X, Y)
ax.set_zlabel('$a*$', fontsize=15, rotation=0)

ax.view_init(elev=10., azim=355)

surf = ax.plot_surface(X, Y, a_star.T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
plt.show()

# X = list(np.arange(x_min, x_min + (N_x + 1) * Delta_x, Delta_x))
# V = list(np.arange(0, (N_v + 1) * Delta_v, Delta_v))
# Data, label = [], []
# for x in range(N_x + 1):
#     for v in range(N_v + 1):
#         for t in range(T + 1):
#             Data.append([x_min + x * Delta_x, v * Delta_v, t])
#             label.append(A_STAR[x, v, t])
# clf = tree.DecisionTreeRegressor()
# clf = clf.fit(Data, label)
# fig = plt.figure()
# r = export_text(clf)
# print(r)
# tree.plot_tree(clf)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graphviz.Source(dot_data)
# fig.savefig("decistion_tree.pdf")
# plt.show()