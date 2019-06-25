# Figure 6.5
# Squared bias, variance, and test mean squared error for ridge regression
# predictions plotted, in the left panel, as a function of lambda (alpha in
# sklearn).  In the right panel, plotted as a function of ratio of l-2 norm
# of ridge beta and ols beta for different values of lambda (alpha in sklearn).

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt

# Training and test data parameters
num_features = 30
train_size = 50
test_size = 200
cor = 0.7
error_sd = 2.0
num_trials = 20
np.random.seed(911)

cov_mat = np.diag(np.ones(num_features) * (1 - cor))
cov_mat = cov_mat + cor

X_test = np.random.multivariate_normal(
    mean=np.zeros(num_features), cov=cov_mat, size=test_size)
y_test = X_test.sum(axis=1) + \
    np.random.normal(loc=0, scale=error_sd, size=test_size)

alpha_vals = np.logspace(-3, 2, 31)  # lambda in ISLR
mse = []
var = []
bias_sq = []
ridge_models = []
beta_ratio = []

for alpha in alpha_vals:
    model = Ridge(alpha=alpha)
    lm_model = LinearRegression()
    y_predict = np.zeros((test_size, num_trials))
    beta_ridge_linear = []
    for j in range(num_trials):
        # Simulate training data
        X_train = np.random.multivariate_normal(
            mean=np.zeros(num_features), cov=cov_mat, size=train_size)
        y_train = X_train.sum(axis=1) + \
            np.random.normal(loc=0, scale=error_sd, size=train_size)
        model.fit(X_train, y_train)  # Fit ridge model
        y_predict[:, j] = model.predict(X_test)
        lm_model.fit(X_train, y_train)  # Fit linear model
        beta_ridge_el2 = np.sqrt(np.sum(model.coef_ ** 2))
        beta_ols_el2 = np.sqrt(np.sum(lm_model.coef_ ** 2))
        beta_ridge_linear.append(beta_ridge_el2 / beta_ols_el2)

    y_predict_mean = y_predict.mean(axis=1)
    var_vec = ((y_predict - y_predict_mean[:, np.newaxis]) ** 2).mean(axis=1)
    bias_sq_vec = (y_predict_mean - y_test) ** 2
    error_sq_vec = ((y_test[:, np.newaxis] - y_predict) ** 2).mean(axis=1)
    mse.append(np.mean(error_sq_vec))
    var.append(np.mean(var_vec))
    bias_sq.append(np.mean(bias_sq_vec))
    beta_ratio.append(np.mean(beta_ridge_linear))
    ridge_models.append(model)


x_vars = {1: alpha_vals, 2: beta_ratio}
fig = plt.figure(figsize=(8, 4))
for k in x_vars.keys():
    ax = fig.add_subplot(1, 2, k)
    x = x_vars[k]
    ax.plot(x, bias_sq, c='k', linestyle=':', label=r'$bias^2$', alpha=0.7)
    ax.plot(x, var, c='g', label='var', alpha=0.7)
    ax.plot(x, mse, c='r', label='mse', linestyle='--', alpha=0.7)
    ax.axhline(y=error_sd ** 2, color='grey', alpha=0.7)
    min_ind = np.argmin(mse)
    ax.scatter(x[min_ind], mse[min_ind], marker='x', c='k', s=100)
    ax.legend()
    ax.set_ylabel('Mean Squared Error')
    if k == 1:
        ax.set_xscale('log')
        ax.set_xlabel(r'$\lambda$')
    if k == 2:
        ax.set_xlabel(
            r'$\|| \hat{\beta}_\lambda^R \||_2 / \|| \hat{\beta} \||_2$')

fig.tight_layout()
