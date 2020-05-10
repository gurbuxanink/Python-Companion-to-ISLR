# hittersRegTree.py
# Code to plot figure 8.1

from statsmodels import datasets
from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hitters = datasets.get_rdataset('Hitters', 'ISLR').data

hitters_use = hitters[['Hits', 'Years', 'Salary']].copy()
hitters_use.dropna(how='any', inplace=True)

tree = DecisionTreeRegressor(max_depth=2)
X = hitters_use[['Hits', 'Years']]
y = np.log(hitters_use['Salary'])

tree.fit(X, y)

fig, ax = plt.subplots()
plot_tree(tree, ax=ax)
