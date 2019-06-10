# Figure 5.4

import matplotlib.pyplot as plt
import numpy as np
from statsmodels import datasets
from numpy.polynomial.polynomial import polyfit, polyval

auto = datasets.get_rdataset('Auto', 'ISLR').data

deg_loocv = []
mse_loocv = []

for p in range(1, 11):
    mse_p = []
    for i in range(auto.shape[0]):
        test_ind = i
        train_ind = np.vectorize(lambda x: x != i)(list(range(auto.shape[0])))
        auto_train = auto.iloc[train_ind]
        auto_test = auto.iloc[test_ind]
        poly_fit = polyfit(auto_train['horsepower'], auto_train['mpg'],
                           deg=p)
        mpg_test = polyval(auto_test['horsepower'], poly_fit)
        mse_p.append((mpg_test - auto_test['mpg']) ** 2)

    mse_loocv.append(np.sum(mse_p) / auto.shape[0])
    deg_loocv.append(p)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.plot(deg_loocv, mse_loocv, marker='o', c='b')

ax2 = fig.add_subplot(122)
np.random.seed(911)

n_folds = 10
# Repeat k-fold cross validation 9 times
for k in range(9):
    # A new split of data in n folds
    fold_ind = np.random.choice(n_folds, size=auto.shape[0])
    mse_allfolds = []

    mse_fold = {}
    for p in range(1, 11):
        mse_fold[p] = []

    for i in range(n_folds):
        train_df = auto.loc[fold_ind != i]
        test_df = auto.loc[fold_ind == i]
        for p in range(1, 11):
            poly_fit = polyfit(train_df['horsepower'],
                               train_df['mpg'], deg=p)
            mpg_test = polyval(test_df['horsepower'], poly_fit)
            mse = np.mean((mpg_test - test_df['mpg']) ** 2)
            mse_fold[p].append(mse)

    for p in mse_fold.keys():
        mse_allfolds.append(np.mean(mse_fold[p]))

    ax2.plot(mse_fold.keys(), mse_allfolds, lw=0.7)

for ax in fig.axes:
    ax.set_xlabel('Degree of Polynomial')
    ax.set_ylabel('Mean Squared Error')
    ax.set_ylim(15, 28)

ax1.set_title('LOOCV')
ax2.set_title('10-fold CV')
fig.tight_layout()
