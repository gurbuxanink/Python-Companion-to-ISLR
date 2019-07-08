# Figure 6.24
# Boxplots of test MSE versus degrees of freedom in lasso
# Training data has 100 observations and 20 features associated with response
# p = 50 and p = 2000 include additional features that are not related to
# response

import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(911)
train_size = 100
test_size = 500
p_true = 20
p_choices = [20, 50, 2000]
error_sd = 2.0
num_trials = 20
cor = 0.7
cov_mat = np.diag(np.zeros(p_true) + 1 - cor)
cov_mat = cov_mat + cor

X_train = np.random.multivariate_normal(mean=np.zeros(p_true),
                                        cov=cov_mat, size=train_size)
y_train = X_train.sum(axis=1) + np.random.normal(loc=0, scale=error_sd,
                                                 size=train_size)

fig = plt.figure(figsize=(8, 3))

# Left panel, p=20
alpha_vals = [12, 5, 1, 0.1]
mse = {}
for alpha in alpha_vals:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    dof = np.sum(np.abs(lasso.coef_) > 1e-8)
    mse[dof] = []
    for i in range(num_trials):
        X_test = np.random.multivariate_normal(mean=np.zeros(p_true),
                                               cov=cov_mat, size=test_size)
        y_test = X_test.sum(axis=1) + np.random.normal(loc=0, scale=error_sd,
                                                       size=test_size)
        y_fit = lasso.predict(X_test)
        error = np.mean((y_fit - y_test) ** 2)
        mse[dof].append(error)

ax1 = fig.add_subplot(131)
pd.DataFrame(mse).boxplot(grid=False, ax=ax1)
ax1.set_title(r'$p = 20$')

# Center panel
p = 50
X_add = np.random.normal(size=(train_size, p - p_true))
X_train = np.concatenate((X_train, X_add), axis=1)
# alpha_vals = [12, 1, 0.01]
mse = {}
for alpha in alpha_vals:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    dof = np.sum(np.abs(lasso.coef_) > 1e-8)
    mse[dof] = []
    for i in range(num_trials):
        X_test = np.random.multivariate_normal(
            mean=np.zeros(p_true), cov=cov_mat, size=test_size)
        y_test = X_test.sum(axis=1) + np.random.normal(loc=0, scale=error_sd,
                                                       size=test_size)
        X_test_add = np.random.normal(size=(test_size, p - p_true))
        X_test = np.concatenate((X_test, X_test_add), axis=1)
        y_fit = lasso.predict(X_test)
        error = np.mean((y_fit - y_test) ** 2)
        mse[dof].append(error)

ax2 = fig.add_subplot(132)
pd.DataFrame(mse).boxplot(grid=False, ax=ax2)
ax2.set_title(r'$p = 50$')

# Right panel
p = 2000
X_add = np.random.normal(size=(train_size, p - X_train.shape[1]))
X_train = np.concatenate((X_train, X_add), axis=1)
# alpha_vals = [11, 0.6, 0.025]
mse = {}
for alpha in alpha_vals:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    dof = np.sum(np.abs(lasso.coef_) > 1e-8)
    mse[dof] = []
    for i in range(num_trials):
        X_test = np.random.multivariate_normal(
            mean=np.zeros(p_true), cov=cov_mat, size=test_size)
        y_test = X_test.sum(axis=1) + np.random.normal(loc=0, scale=error_sd,
                                                       size=test_size)
        X_test_add = np.random.normal(size=(test_size, p - p_true))
        X_test = np.concatenate((X_test, X_test_add), axis=1)
        y_fit = lasso.predict(X_test)
        error = np.mean((y_fit - y_test) ** 2)
        mse[dof].append(error)

ax3 = fig.add_subplot(133)
pd.DataFrame(mse).boxplot(grid=False, ax=ax3)
ax3.set_title(r'$p = 2000$')

for ax in fig.axes:
    ax.set_xlabel('Degrees of Freedom')

fig.tight_layout()
