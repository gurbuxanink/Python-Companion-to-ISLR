# Plot figure 3.16

import numpy as np
from sklearn import neighbors, datasets
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(911)

x1 = np.random.uniform(0, 5, 64)
x2 = np.random.uniform(0, 5, 64)
y = x1 + x2 + np.random.normal(size=x1.size)
X = np.column_stack((x1, x2))

knn1 = neighbors.KNeighborsRegressor(n_neighbors=1)
knn1.fit(X, y)
y_fit = knn1.predict(X)

x1_xx = np.linspace(x1.min(), x1.max())
x2_xx = np.linspace(x2.min(), x2.max())
x1_grid, x2_grid = np.meshgrid(x1_xx, x2_xx)
y_pred = knn1.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
y_grid = np.reshape(y_pred, x1_grid.shape)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_wireframe(x1_grid, x2_grid, y_grid, alpha=0.4, rcount=15, ccount=15)
ax1.scatter(x1, x2, y, color='brown', alpha=0.6)
for i in range(x1.size):
    ax1.plot([x1[i], x1[i]], [x2[i], x2[i]], [y_fit[i], y[i]], color='brown',
             alpha=0.6)

ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.set_zlabel(r'$y$')
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False


knn9 = neighbors.KNeighborsRegressor(n_neighbors=9)
knn9.fit(X, y)
y_fit9 = knn9.predict(X)

y_pred9 = knn9.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
y_grid9 = np.reshape(y_pred9, x1_grid.shape)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_wireframe(x1_grid, x2_grid, y_grid9, alpha=0.4, rcount=15, ccount=15)
ax2.scatter(x1, x2, y, color='brown', alpha=0.6)
for i in range(x1.size):
    ax2.plot([x1[i], x1[i]], [x2[i], x2[i]], [y_fit6[i], y[i]], color='brown',
             alpha=0.6)

ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')
ax2.set_zlabel(r'$y$')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

fig.tight_layout()
