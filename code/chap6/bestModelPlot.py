# Figure 6.2
# Plot Cp, BIC, and ajdusted r-squared for best model for each count of
# variables.  Use all subsets of explanatory variables for Credit data.

import numpy as np
import matplotlib.pyplot as plt
import subsetSelection
from statsmodels import datasets
from subsetSelection import allSubsets, bestSubset, C_p

credit = datasets.get_rdataset('Credit', 'ISLR').data
credit.drop(columns='ID', inplace=True)

y_var = 'Balance'
x_var = list(credit.columns)
x_var.remove(y_var)

all_models = subsetSelection.allSubsets(y_var, x_var, credit)

count_var_cp, best_model_cp, best_vars_cp, best_cp = \
    bestSubset(all_models, C_p, False, 'func')

count_var_bic, best_model_bic, best_vars_bic, best_bic = \
    bestSubset(all_models, 'bic', False)

count_var_adjrsq, best_model_adjrsq, best_vars_adjrsq, best_adjrsq = \
    bestSubset(all_models, 'rsquared_adj')

fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(131)
ax1.scatter(count_var_cp, best_cp, c='b', s=20, alpha=0.7)
ax1.plot(count_var_cp, best_cp, c='r', linewidth=1, alpha=0.7)
best_cp_ind = np.argmin(best_cp)
ax1.scatter(count_var_cp[best_cp_ind], best_cp[best_cp_ind],
            marker='x', s=100, c='g')
ax1.set_ylabel(r'$C_p$')

ax2 = fig.add_subplot(132)
ax2.scatter(count_var_bic, best_bic, c='b', s=20, alpha=0.7)
ax2.plot(count_var_bic, best_bic, c='r', linewidth=1, alpha=0.7)
best_bic_ind = np.argmin(best_bic)
ax2.scatter(count_var_bic[best_bic_ind], best_bic[best_bic_ind], marker='x',
            s=100, c='g')
ax2.set_ylabel('BIC')

ax3 = fig.add_subplot(133)
ax3.scatter(count_var_adjrsq, best_adjrsq, c='b', s=20, alpha=0.7)
ax3.plot(count_var_adjrsq, best_adjrsq, c='r', linewidth=1, alpha=0.7)
best_adjrsq_ind = np.argmax(best_adjrsq)
ax3.scatter(count_var_adjrsq[best_adjrsq_ind],
            best_adjrsq[best_adjrsq_ind], marker='x', s=100, c='g')
ax3.set_ylabel(r'$Adjusted \; R^2$')

for ax in fig.axes:
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xticklabels([2, 4, 6, 8, 10])
    ax.set_xlabel('Number of Predictors')

fig.tight_layout()
