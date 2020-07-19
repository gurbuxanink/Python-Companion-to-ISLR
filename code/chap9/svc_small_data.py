# svc_small_data.py
# Code to plot figure 9.6

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=15, n_features=2, centers=2,
                  cluster_std=4, random_state=0)


x1_min, x2_min = X.min(axis=0)
x1_max, x2_max = X.max(axis=0)
x1 = np.linspace(x1_min, x1_max)
x2 = np.linspace(x2_min, x2_max)
x1_grid, x2_grid = np.meshgrid(x1, x2)
X_array = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

my_colors = ['red', 'blue']

svc1 = SVC(C=0.25, kernel='linear')
svc1.fit(X[:12], y[:12])
z1 = svc1.decision_function(X_array)
z1_grid = z1.reshape(x1_grid.shape)

svc = SVC(C=0.25, kernel='linear')
svc.fit(X, y)
z = svc.decision_function(X_array)
z_grid = z.reshape(x1_grid.shape)

fig, ax = plt.subplots(1, 2)

for i in range(12):
    ax[0].scatter(X[i, 0], X[i, 1], c=my_colors[y[i]], alpha=0.7, s=50,
                  marker='$' + str(i) + '$')

ax[0].contour(x1_grid, x2_grid, z1_grid,
              levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='gray')

for i in range(15):
    ax[1].scatter(X[i, 0], X[i, 1], c=my_colors[y[i]], alpha=0.7, s=50,
                  marker='$' + str(i) + '$')

ax[1].contour(x1_grid, x2_grid, z_grid, levels=[-1, 0, 1],
              linestyles=['--', '-', '--'], colors='gray')

for axi in ax:
    axi.set(xlabel=r'$X_1$', ylabel=r'$X_2$')

fig.tight_layout()
