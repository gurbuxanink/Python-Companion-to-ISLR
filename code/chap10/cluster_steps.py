# cluster_steps.py
# Plot figures 10.10 and 10.11

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pdb

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


def include_ind(z_ind, Z):
    '''Find all indexes included in z_ind'''
    if int(Z[z_ind, 3]) == 2:
        return [int(id) for id in Z[z_ind, :2]]
    else:
        x_ind = Z[z_ind, :2]
        use_ind = []
        for obs in x_ind:
            if obs < Z[-1, -1]:
                use_ind.append(int(obs))
            else:
                new_ind = int(obs) - int(Z[-1, -1])
                use_ind.extend(include_ind(new_ind, Z))
        return use_ind


# Figure 10.11
fig_steps, ax = plt.subplots(2, 2)

my_colors = ['red', 'green', 'gray']
for i, axi in enumerate(ax.flatten()):
    used_ind = []
    for j in np.arange(i)[::-1]:
        plot_ind = include_ind(j, Z)
        for ind in plot_ind:
            if ind not in used_ind:
                axi.text(X[ind, 0], X[ind, 1], str(y[ind]), ha='center',
                         va='center', bbox=dict(facecolor=my_colors[j], alpha=0.2))
        used_ind.extend(plot_ind)
    remain_ind = list(set(range(len(y))) - set(used_ind))
    for ind in remain_ind:
        axi.text(X[ind, 0], X[ind, 1], str(y[ind]), ha='center', va='center')
    axi.set(xlim=[x1_min - 0.2, x1_max + 0.2],
            ylim=[x2_min - 0.2, x2_max + 0.2],
            xlabel=r'$X_1$', ylabel=r'$X_2$')

fig_steps.tight_layout()
