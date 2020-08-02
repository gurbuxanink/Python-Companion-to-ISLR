# Convenience functions for svm

import numpy as np
import matplotlib.pyplot as plt


def svm_model_plot(model, X, y, ax=None, plot_support_vec=True):
    '''model is a fitted model of class sklearn.SVC
    X is n x 2 features matrix, y is n x 1 class array
    Returns a graph with scatter plot of X, support vectors of model,
    and svc hyperplane and margins'''
    # import numpy as np

    x1_min, x2_min = X.min(axis=0)
    x1_max, x2_max = X.max(axis=0)
    x1_array = np.linspace(x1_min, x1_max)
    x2_array = np.linspace(x2_min, x2_max)
    x1_grid, x2_grid = np.meshgrid(x1_array, x2_array)

    z = model.decision_function(
        np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T)
    z_grid = z.reshape(x1_grid.shape)

    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('RdBu', 2),
               alpha=0.7)
    if plot_support_vec:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   marker='x', s=100, c='black', alpha=0.7)
    ax.contour(x1_grid, x2_grid, z_grid, levels=[-1, 0, 1],
               linestyles=['--', '-', '--'], colors='black', alpha=0.7)
    ax.set(xlabel=r'$X_1$', ylabel=r'$X_2$')
    return ax
