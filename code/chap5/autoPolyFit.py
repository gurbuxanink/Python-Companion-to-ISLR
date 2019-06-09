# Figure 5.2

import numpy as np
from statsmodels import datasets
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit, polyval


auto = datasets.get_rdataset('Auto', 'ISLR').data
n_obs = auto.shape[0]
n_train = int(n_obs / 2)
n_test = n_obs - n_train

np.random.seed(911)
train_ind = np.random.choice(range(n_obs), n_train, replace=False)
test_ind = set(range(n_obs)).difference(set(train_ind))
test_ind = list(test_ind)
auto_train = auto.iloc[train_ind]
auto_test = auto.iloc[test_ind]

poly_degree = []
poly_mse = []
for p in range(1, 11):
    poly_fit = polyfit(auto_train['horsepower'], auto_train['mpg'], deg=p)
    mpg_test = polyval(auto_test['horsepower'], poly_fit)
    mse_test = np.mean((mpg_test - auto_test['mpg']) ** 2)
    poly_degree.append(p)
    poly_mse.append(mse_test)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.plot(poly_degree, poly_mse, marker='o', c='r')
ax1.set_xlabel('Degree of Polynomial')
ax1.set_ylabel('Mean Squared Error')

ax2 = fig.add_subplot(122)
for i in range(10):
    train_ind = np.random.choice(range(n_obs), n_train, replace=False)
    test_ind = set(range(n_obs)).difference(set(train_ind))
    test_ind = list(test_ind)
    auto_train = auto.iloc[train_ind]
    auto_test = auto.iloc[test_ind]

    poly_degree = []
    poly_mse = []
    for p in range(1, 11):
        poly_fit = polyfit(auto_train['horsepower'], auto_train['mpg'], deg=p)
        mpg_test = polyval(auto_test['horsepower'], poly_fit)
        mse_test = np.mean((mpg_test - auto_test['mpg']) ** 2)
        poly_degree.append(p)
        poly_mse.append(mse_test)

    ax2.plot(poly_degree, poly_mse, linewidth=1)

ax2.set_xlabel('Degree of Polynomial')
ax2.set_ylabel('Mean Squared Error')
fig.tight_layout()
