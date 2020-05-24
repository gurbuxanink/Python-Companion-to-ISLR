# hittersRegTree.py
# Code to plot figures 8.1 and 8.2
# How to find region boundaries (e.g., 4.5, 117.5) from tree estimator?

from statsmodels import datasets
from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

hitters = datasets.get_rdataset('Hitters', 'ISLR').data

hitters_use = hitters[['Hits', 'Years', 'Salary']].copy()
hitters_use.dropna(how='any', inplace=True)

tree = DecisionTreeRegressor(max_depth=2)
X = hitters_use[['Hits', 'Years']]
y = np.log(hitters_use['Salary'])

tree.fit(X, y)

fig, ax = plt.subplots()
plot_tree(tree, feature_names=['Hits', 'Years'], ax=ax)

# Plot decision tree regions and training data
xx = np.linspace(hitters_use['Years'].min(), hitters_use['Years'].max())
yy = np.linspace(hitters_use['Hits'].min(), hitters_use['Hits'].max())
x_grid, y_grid = np.meshgrid(xx, yy)
zz = tree.predict(np.vstack((y_grid.ravel(), x_grid.ravel())).T)
z_grid = zz.reshape(x_grid.shape)

fig, ax = plt.subplots()
ax.contour(x_grid, y_grid, z_grid, 2, colors='black')
hitters_use.plot(x='Years', y='Hits', kind='scatter', alpha=0.7, ax=ax)
