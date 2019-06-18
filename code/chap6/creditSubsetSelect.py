# Figure 6.1
# Subset selection on Credit data

from statsmodels import datasets
import statsmodels.formula.api as smf
# import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

credit = datasets.get_rdataset('Credit', 'ISLR').data
credit = credit.drop(columns='ID')

all_models = {}

all_x = list(credit.columns)
y_var = 'Balance'
all_x.remove(y_var)

# All subsets
for p in range(1, len(all_x) + 1):
    all_models[p] = []
    x_combinations = combinations(all_x, p)
    for select_var in x_combinations:
        my_formula = y_var + ' ~ ' + ' + '.join(select_var)
        lm_model = smf.ols(my_formula, data=credit)
        lm_fit = lm_model.fit()
        all_models[p].append(lm_fit)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
model_degree = []
best_model_rss = []
best_model_rsquared = []
for p in all_models.keys():
    min_rss = np.inf
    max_rsquared = 0
    for model in all_models[p]:
        ax1.scatter(p, model.ssr, c='b', s=10, alpha=0.7)
        ax2.scatter(p, model.rsquared, c='b', s=10, alpha=0.7)
        min_rss = min(min_rss, model.ssr)
        max_rsquared = max(max_rsquared, model.rsquared)
    best_model_rss.append(min_rss)
    best_model_rsquared.append(max_rsquared)
    model_degree.append(p)
ax1.plot(model_degree, best_model_rss, c='r')
ax2.plot(model_degree, best_model_rsquared, c='r')
ax1.set_xlabel('Number of Predictors')
ax1.set_ylabel('Residual Sum of Squares')
ax2.set_xlabel('Number of Predictors')
ax2.set_ylabel(r'$R^2$')

fig.tight_layout()
