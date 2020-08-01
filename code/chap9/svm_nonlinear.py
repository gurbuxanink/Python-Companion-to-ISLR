# svm_nonlinear.py
# Code to plot figure 9.9

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV

np.random.seed(0)
my_df = pd.DataFrame({'x1': np.random.normal(loc=0, scale=2, size=50)})
my_df['x2'] = my_df['x1'] + np.random.normal(size=50)
my_df['y'] = my_df.apply(lambda row: 0 if np.sum(row ** 2) < 8 else 1, axis=1)

x1_min, x1_max = my_df['x1'].agg([np.min, np.max])
x2_min, x2_max = my_df['x2'].agg([np.min, np.max])
x1_array = np.linspace(x1_min, x1_max)
x2_array = np.linspace(x2_min, x2_max)
x1_grid, x2_grid = np.meshgrid(x1_array, x2_array)

svm = SVC(C=25, kernel='poly', degree=3)
svm.fit(my_df[['x1', 'x2']], my_df['y'])

z = svm.decision_function(np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T)
z_grid = z.reshape(x1_grid.shape)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(my_df['x1'], my_df['x2'], c=my_df['y'],
            cmap=plt.cm.get_cmap('RdBu', 2), alpha=0.7)
ax1.contour(x1_grid, x2_grid, z_grid, levels=[-1, 0, 1],
            linestyles=['--', '-', '--'], colors='black', alpha=0.7)

svm2 = SVC(kernel='rbf')
svm2.fit(my_df[['x1', 'x2']], my_df['y'])

z = svm2.decision_function(np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T)
z_grid = z.reshape(x1_grid.shape)

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(my_df['x1'], my_df['x2'], c=my_df['y'],
            cmap=plt.cm.get_cmap('RdBu', 2), alpha=0.7)
ax2.contour(x1_grid, x2_grid, z_grid, levels=[-1, 0, 1],
            linestyles=['--', '-', '--'], colors='black', alpha=0.7)

for ax in fig.axes:
    ax.set(xlabel=r'$X_1$', ylabel=r'$X_2$')

fig.tight_layout()

# svm = SVC(kernel='poly')
# param_grid = {'C': [0.0001, 0.001, 0.01], 'gamma': [1, 10, 100, 1000]}
# grid = GridSearchCV(svm, param_grid=param_grid)
# grid.fit(my_df[['x1', 'x2']], my_df['y'])
