# Figure 6.3
# Select best model using BIC, validation set error, and k-fold
# cross validation error.  For best models, plot error versus number of
# predictors in three panels.

from statsmodels import datasets
from subsetSelection import allSubsets, bestSubset, bestSubsetCrossVal,\
    bestSubsetValidation
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('code/chap6/')


credit = datasets.get_rdataset('Credit', 'ISLR').data
credit.drop(columns='ID', inplace=True)

y_var = 'Balance'
x_vars = list(credit.columns)
x_vars.remove(y_var)

all_models = allSubsets(y_var, x_vars, credit)
num_vars_bic, best_models_bic, best_models_vars_bic, best_models_bic = \
    bestSubset(all_models, 'bic', metric_max=False)
best_models_sqrt_bic = [np.sqrt(x) for x in best_models_bic]

best_models_validation = bestSubsetValidation(y_var, x_vars, credit)

best_models_kfold = bestSubsetCrossVal(y_var, x_vars, credit)

fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(131)
ax1.scatter(num_vars_bic, best_models_sqrt_bic, c='b', s=20, alpha=0.7)
ax1.plot(num_vars_bic, best_models_sqrt_bic, c='r', linewidth=1,
         alpha=0.7)
best_bic_ind = np.argmin(best_models_sqrt_bic)
ax1.scatter(num_vars_bic[best_bic_ind], best_models_sqrt_bic[best_bic_ind],
            marker='x', s=100, c='g')
ax1.set_ylabel('Square Root of BIC')

ax2 = fig.add_subplot(132)
num_vars_validation = [p for p in best_models_validation.keys()]
mse_validation = [best_models_validation[p]['mse']
                  for p in best_models_validation.keys()]
ax2.scatter(num_vars_validation, mse_validation, c='b', s=20, alpha=0.7)
ax2.plot(num_vars_validation, mse_validation, c='r', linewidth=1,
         alpha=0.7)
best_val_ind = np.argmin(mse_validation)
ax2.scatter(num_vars_validation[best_val_ind], mse_validation[best_val_ind],
            marker='x', s=100, c='g')
ax2.set_ylabel('Validation Set Error')

ax3 = fig.add_subplot(133)
num_vars_kfold = [p for p in best_models_kfold.keys()]
mse_kfold = [best_models_kfold[p]['mse'] for p in best_models_kfold.keys()]
ax3.scatter(num_vars_kfold, mse_kfold, c='b', s=20, alpha=0.7)
ax3.plot(num_vars_kfold, mse_kfold, c='r', linewidth=1, alpha=0.7)
best_kfold_ind = np.argmin(mse_kfold)
ax3.scatter(num_vars_kfold[best_kfold_ind], mse_kfold[best_kfold_ind],
            marker='x', s=100, c='g')
ax3.set_ylabel('Cross-Validation Error')

for ax in fig.axes:
    for tick in ax.get_yticklabels():
        tick.set_rotation(60)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xticklabels([2, 4, 6, 8, 10])
    ax.set_xlabel('Number of Predictors')

fig.tight_layout()
