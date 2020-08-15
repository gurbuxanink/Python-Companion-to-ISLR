# cluster_steps.py
# Plot figures 10.10 and 10.11

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

np.random.seed(4)
X = np.random.normal(size=(9, 2))
y = np.arange(1, 10)
x1_min, x2_min = X.min(axis=0)
x1_max, x2_max = X.max(axis=0)

Z = linkage(X, method='ward')

# Figure 10.10
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1, 2, 1)
dendrogram(Z, labels=y,  ax=ax1)

ax2 = fig.add_subplot(1, 2, 2)
for i, s in enumerate(y):
    ax2.text(X[i, 0], X[i, 1], str(s), ha='center', va='center')
ax2.set(xlim=[x1_min - 0.2, x1_max + 0.2], ylim=[x2_min - 0.2, x2_max + 0.2],
        xlabel=r'$X_1$', ylabel=r'$X_2$')

fig.tight_layout()
