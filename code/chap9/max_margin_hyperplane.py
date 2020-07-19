# max_margin_hyperplane.py
# Plot figure 9.3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from lines import line
from sklearn.svm import SVC

X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=0.7,
                  random_state=0)

svc = SVC(C=10, kernel='linear')
svc.fit(X, y)


x1_min, x2_min = X.min(axis=0)
x1_max, x2_max = X.max(axis=0)
x1 = np.linspace(x1_min, x1_max)
x2 = np.linspace(x2_min, x2_max)
x1_grid, x2_grid = np.meshgrid(x1, x2)
z_predict = svc.predict(np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T)
# z_grid = z_grid.reshape(x1_grid.shape)
p_grid = svc.decision_function(np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T)
p_grid = p_grid.reshape(x1_grid.shape)


fig = plt.figure(figsize=(x1_max - x1_min, x2_max - x2_min))
ax = fig.add_subplot()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('RdBu', 2), alpha=0.7)
ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1],
           marker='x', s=100, color='black', alpha=0.7)
ax.scatter(x1_grid.ravel(), x2_grid.ravel(), c=z_predict, cmap=plt.cm.get_cmap('RdBu', 2),
           marker='.', s=10, alpha=0.3)
ax.contour(x1_grid, x2_grid, p_grid, levels=[-1, 0, 1],  linestyles=['--', '-', '--'],
           colors='gray')

# With well-spearated clusters and linear hyperplane,
# there will usually be three support vectors
# Find the two in the same class
# This will not work if there are only two support vectors
if svc.support_vectors_.shape[0] == 3:
    classes, counts = np.unique(svc.predict(
        svc.support_vectors_), return_counts=True)

    class_with_2_sv = np.where(counts == 2)[0][0]
    sv_pair_ind = np.where(svc.predict(
        svc.support_vectors_) == class_with_2_sv)[0]
    sv_alone_ind = np.where(svc.predict(
        svc.support_vectors_) != class_with_2_sv)[0]

    line1 = line()
    line1.from_two_points(svc.support_vectors_[
        sv_pair_ind[0]], svc.support_vectors_[sv_pair_ind[1]])
    line2 = line()
    line2.from_slope_point(line1.slope_, svc.support_vectors_[sv_alone_ind[0]])
    mid_line = line()
    mid_line.from_slope_intercept(
        line1.slope_, 0.5 * (line1.intercept_ + line2.intercept_))

    arrow_slope = -1 / line1.slope_  # perpendicular to hyperplane

    for point in svc.support_vectors_:
        arrow_start = point
        arrow_line = line()
        arrow_line.from_slope_point(arrow_slope, arrow_start)
        arrow_end = mid_line.get_intersection(arrow_line)
        arrow = np.vstack([arrow_start, arrow_end])
        ax.plot(arrow[:, 0], arrow[:, 1], color='gray')

    ax.axis('equal')

ax.set(xlabel=r'$X_1$', ylabel=r'$X_2$')
