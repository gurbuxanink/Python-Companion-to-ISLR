# km_cluster.py
# Code to plot figures 10.5, 10.6, and 10.7

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=60, n_features=2, centers=3, random_state=0)

# Figure 10.5
# Plot results of k-means clustering for k=2, 3, 4
fig = plt.figure(figsize=(9, 4))
for i, k_clusters in enumerate([2, 3, 4]):
    kmeans = KMeans(n_clusters=k_clusters, random_state=0)
    kmeans.fit(X)
    ax = fig.add_subplot(1, 3, i+1)
    ax.scatter(X[:, 0], X[:, 1], c=kmeans.predict(
        X), cmap=plt.cm.get_cmap('viridis', k_clusters), alpha=0.7)
    ax.set(title='K = ' + str(k_clusters))

fig.tight_layout()

# Figure 10.6
# Illustrate steps in k-means clustering algorithm


def find_clusters(X, k_clusters, in_seed=0):
    np.random.seed(in_seed)
    y_iter = np.random.choice(np.arange(k_clusters),
                              size=X.shape[0], replace=True)
    min_sum_dist_sq = 1e10
    sum_dist_sq = 1e9
    y_dict = {}
    num_iter = 0
    y_dict[num_iter] = y_iter
    dist_dict = {}
    centroid_dict = {}
    while sum_dist_sq < min_sum_dist_sq:  # Distance is still decreasing
        min_sum_dist_sq = sum_dist_sq
        centroids = {}
        centroid_dist_sq = {}
        for yi in np.arange(k_clusters):
            centroids[yi] = X[y_iter == yi, :].mean(axis=0)
            centroid_dist_sq[yi] = (
                (X[y_iter == yi, :] - centroids[yi]) ** 2).sum()
        sum_dist_sq = sum([centroid_dist_sq[yi] for yi in centroid_dist_sq])
        dist_dict[num_iter] = sum_dist_sq
        centroid_dict[num_iter] = centroids

        X_dist_sq = np.zeros([X.shape[0], k_clusters])
        for center in centroids:
            X_dist_sq[:, center] = ((X - centroids[center]) ** 2).sum(axis=1)
        y_iter = X_dist_sq.argmin(axis=1)
        num_iter += 1
        y_dict[num_iter] = y_iter

    return dict(num_iter=num_iter, y_dict=y_dict, dist_dict=dist_dict,
                centroids=centroid_dict)


def centroid_plot(centroids, ax=None):
    '''Plot centroids on ax.  Assumes centroids is a dictionary of points.'''
    if ax is None:
        ax = plt.gca()
    for center in centroids:
        ax.scatter(centroids[center][0], centroids[center][1], marker='x',
                   color='black', s=100)


cluster_res = find_clusters(X, 3)

fig_kmeans_steps = plt.figure()
ax1 = fig_kmeans_steps.add_subplot(2, 3, 1)
ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
ax1.set(title='Data')

ax2 = fig_kmeans_steps.add_subplot(2, 3, 2)
ax2.scatter(X[:, 0], X[:, 1], c=cluster_res['y_dict'][0],
            cmap=plt.cm.get_cmap('viridis', 3), alpha=0.7)
ax2.set(title='Step 1')

ax3 = fig_kmeans_steps.add_subplot(2, 3, 3)
ax3.scatter(X[:, 0], X[:, 1], c=cluster_res['y_dict'][0],
            cmap=plt.cm.get_cmap('viridis', 3), alpha=0.7)
centroid_plot(cluster_res['centroids'][0], ax3)
ax3.set(title='Iteration 1, Step 2a')

ax4 = fig_kmeans_steps.add_subplot(2, 3, 4)
ax4.scatter(X[:, 0], X[:, 1], c=cluster_res['y_dict'][1],
            cmap=plt.cm.get_cmap('viridis', 3), alpha=0.7)
centroid_plot(cluster_res['centroids'][0], ax4)
ax4.set(title='Iteration 1, Step 2b')

ax5 = fig_kmeans_steps.add_subplot(2, 3, 5)
ax5.scatter(X[:, 0], X[:, 1], c=cluster_res['y_dict'][1],
            cmap=plt.cm.get_cmap('viridis', 3), alpha=0.7)
centroid_plot(cluster_res['centroids'][1], ax5)
ax5.set(title='Iteration 2, Step 2a')

ax6 = fig_kmeans_steps.add_subplot(2, 3, 6)
ax6.scatter(X[:, 0], X[:, 1], c=cluster_res['y_dict'][cluster_res['num_iter']],
            cmap=plt.cm.get_cmap('viridis', 3), alpha=0.7)
centroid_plot(cluster_res['centroids'][cluster_res['num_iter'] - 1], ax6)
ax6.set(title='Final Results')

fig_kmeans_steps.tight_layout()

# Plot K-means clustering results with different starting points
# Figure 10.7
fig_kmeans_seeds, ax = plt.subplots(2, 3)
my_seeds = [0, 21, 33, 45, 56, 67]
cluster_res_list = [find_clusters(X, 3, in_seed) for in_seed in my_seeds]

dist = [cluster_res['dist_dict'][cluster_res['num_iter'] - 1]
        for cluster_res in cluster_res_list]
best_dist = min(dist)

for cluster_res,  axi in zip(cluster_res_list, ax.flatten()):
    # cluster_res = find_clusters(X, 3, in_seed=in_seed)
    axi.scatter(X[:, 0], X[:, 1],
                c=cluster_res['y_dict'][cluster_res['num_iter']],
                cmap=plt.cm.get_cmap('viridis', 3), alpha=0.7)
    if np.allclose(best_dist, cluster_res['dist_dict'][
            cluster_res['num_iter'] - 1]):
        title_color = 'red'
    else:
        title_color = 'gray'

    axi.set_title(label=str(round(cluster_res['dist_dict'][
        cluster_res['num_iter'] - 1], 2)), color=title_color)

fig_kmeans_seeds.tight_layout()
