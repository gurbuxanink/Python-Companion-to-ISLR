# Figure 4.5

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from matplotlib import cm

fig = plt.figure(figsize=(8, 8))
correlations = [0, 0.7]
i_ax = 1
for cor in correlations:
    ax = fig.add_subplot(2, 2, i_ax, projection='3d')
    X = np.random.multivariate_normal(mean=[0, 0],
                                      cov=[[1, cor], [cor, 1]],
                                      size=10000)
    kernel = gaussian_kde(X.T)
    x1_min, x2_min = X.min(axis=0)
    x1_max, x2_max = X.max(axis=0)
    x1_array = np.linspace(x1_min, x1_max)
    x2_array = np.linspace(x2_min, x2_max)
    x1_grid, x2_grid = np.meshgrid(x1_array, x2_array)
    z = kernel.evaluate(np.column_stack((x1_grid.ravel(), x2_grid.ravel())).T)
    z_grid = np.reshape(z.T, x1_grid.shape)
    ax.plot_surface(x1_grid, x2_grid, z_grid, ccount=30, rcount=30,
                    cmap=cm.coolwarm)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')

    # Easier to see correlation in this plot
    # Comment out next group of lines, uncomment this group
    # ax = fig.add_subplot(2, 2, i_ax + 2, projection='3d')
    # ax.contourf(x1_grid, x2_grid, z_grid, cmap=cm.coolwarm)
    # ax.grid(False)
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_xlabel(r'$X_1$')
    # ax.set_ylabel(r'$X_2$')

    ax = fig.add_subplot(2, 2, i_ax + 2)
    ax.contour(x1_grid, x2_grid, z_grid, cmap=cm.coolwarm)
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_xticks([])
    ax.set_yticks([])

    i_ax += 1

fig.tight_layout()
