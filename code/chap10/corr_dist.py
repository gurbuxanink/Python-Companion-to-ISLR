# corr_dist.py
# Code to plot figure 10.13

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

np.random.seed(0)
X = np.random.normal(size=(20, 3))
X[:, :2] += 2
X[:, 2] += X[:, 0] * 3 + 10

fig, ax = plt.subplots()
styles = ['-', '--', ':']
for j in range(3):
    ax.plot(X[:, j], linestyle=styles[j], label='Observation ' + str(j))
ax.legend(frameon=False)
ax.set(xticks=[0, 5, 10, 15, 20])
fig.tight_layout()

# Euclidean distance
print(pdist(X.T).round(2))

# Correlation coefficients
print(np.corrcoef(X, rowvar=False).round(2))
