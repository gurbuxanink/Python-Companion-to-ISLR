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

# Plot figure 10.8
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('viridis', 3),
           alpha=0.7)
ax.set(xlabel=r'$X_1$', ylabel=r'$X_2$')

# Plot figure 10.9
Z = linkage(X, method='ward')

fig_dendro = plt.figure(figsize=(8, 3))

for i, thresh in enumerate([21, 15, 10]):
    ax = fig_dendro.add_subplot(1, 3, i + 1)
    dendrogram(Z, color_threshold=thresh, no_labels=True, ax=ax)
    ax.axhline(y=thresh, linestyle='--', color='black', alpha=0.7)
    ax.set(ylim=[0, 20])

fig_dendro.tight_layout()
