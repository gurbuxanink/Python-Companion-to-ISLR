# dendrogram.py
# Plot figures 10.8 and 10.9

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

np.random.seed(0)
X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 45)
X[15:30, 0] -= 3
X[30:, 1] += 3
y = np.concatenate([np.zeros(15), np.ones(15), np.ones(15) * 2])
y = y.astype(int)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('viridis', 3),
           alpha=0.7)
ax.set(xlabel=r'$X_1$', ylabel=r'$X_2$')
