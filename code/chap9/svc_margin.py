# svc_margin.py
# Code to plot figure 9.7

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import numpy as np

X, y = make_blobs(30, centers=2, cluster_std=1.0, random_state=0)

x1_min, x2_min = X.min(axis=0)
x1_max, x2_max = X.max(axis=0)
x1_array = np.linspace(x1_min, x1_max)
x2_array = np.linspace(x2_min, x2_max)
x1_grid, x2_grid = np.meshgrid(x1_array, x2_array)

c_vals = [0.1, 1, 5, 100]

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
for i, c in enumerate(c_vals):
    axi = ax.flatten()[i]
    svc = SVC(C=c, kernel='linear')
    svc.fit(X, y)
    z = svc.decision_function(np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T)
    z_grid = z.reshape(x1_grid.shape)

    axi.scatter(X[:, 0], X[:, 1], c=y,
                cmap=plt.cm.get_cmap('RdBu', 2), alpha=0.7)
    axi.contour(x1_grid, x2_grid, z_grid, levels=[-1, 0, 1],
                linestyles=['--', '-', '--'], colors='black', alpha=0.7)
    axi.set(xlabel=r'$X_1$', ylabel=r'$X_2$', title=r'$C = ' + str(c) + '$')

fig.tight_layout()
