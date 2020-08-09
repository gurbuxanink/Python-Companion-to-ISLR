# km_cluster.py
# Code to plot figures 10.6, 10.7, and 10.8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=60, n_features=2, centers=3, random_state=0)

fig = plt.figure(figsize=(9, 4))
for i, k_clusters in enumerate([2, 3, 4]):
    kmeans = KMeans(n_clusters=k_clusters, random_state=0)
    kmeans.fit(X)
    ax = fig.add_subplot(1, 3, i+1)
    ax.scatter(X[:, 0], X[:, 1], c=kmeans.predict(
        X), cmap=plt.cm.get_cmap('viridis', k_clusters), alpha=0.7)
    ax.set(title='K = ' + str(k_clusters))

fig.tight_layout()
