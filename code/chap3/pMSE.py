# Plot figure 3.20

import numpy as np
from statsmodels.regression import linear_model
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

def myFunc(x):
    return 2 + 3 * x - 2 * x ** 3

error_sd = 0.2
x_min, x_max = -1.0, 1.0
train_size, test_size = 50, 200
k_range = range(1, 21, 2)
add_var_sd = 0.4

np.random.seed(911)
x_train = np.random.uniform(x_min, x_max, size=train_size)
y_train = np.vectorize(myFunc)(x_train) + \
    np.random.normal(loc=0.0, scale=error_sd, size=train_size)
x_train = add_constant(x_train)

x_test = np.random.uniform(x_min, x_max, size=test_size)
y_test = np.vectorize(myFunc)(x_test) + \
    np.random.normal(loc=0.0, scale=error_sd, size=test_size)
x_test = add_constant(x_test)

ols_reg = linear_model.OLS(y_train, x_train)
ols_fit = ols_reg.fit()
y_pred_ols = ols_fit.predict(exog=x_test)
ols_mse = np.mean((y_pred_ols - y_test) ** 2)

fig = plt.figure()
ax_num = 0
for p in [1, 2, 3, 4, 10, 20]:
    # Add fake variables as needed
    num_add_var = p - (x_train.shape[1] - 1)
    for j in range(num_add_var):
        x_train = np.column_stack(
            (x_train,
             np.random.normal(loc=0, scale=add_var_sd, size=train_size)))
        x_test = np.column_stack(
            (x_test,
             np.random.normal(loc=0, scale=add_var_sd, size=test_size)))

    ols_reg = linear_model.OLS(y_train, x_train)
    ols_fit = ols_reg.fit()
    y_pred_ols = ols_fit.predict(exog=x_test)
    ols_mse = np.mean((y_pred_ols - y_test) ** 2)
    
    k_vals = []
    knn_mse = []
    for k in k_range:
        knn_reg = KNeighborsRegressor(n_neighbors=k)
        knn_reg.fit(x_train[:,1:], y_train)
        y_pred_knn = knn_reg.predict(x_test[:,1:])
        mse = np.mean((y_pred_knn - y_test) ** 2)
        k_vals.append(k)
        knn_mse.append(mse)

    k_inv = [1.0 / k for k in k_vals]
    ax_num += 1
    ax = fig.add_subplot(2, 3, ax_num)
    ax.axhline(y=ols_mse, linestyle='--', c='k')
    ax.plot(k_inv, knn_mse, linestyle='-.', c='g')
    ax.set_ylim(0, 1)
    ax.set_xscale('log')
    ax.set_xticks([0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels([0.1, 0.2, 0.5, 1.0])
    ax.set_title('p = ' + str(p))

    if ax_num in [1, 4]:
        ax.set_ylabel('Mean Squared Error')
    if ax_num in [4, 5, 6]:
        ax.set_xlabel(r'$\frac{1}{K}$')

fig.tight_layout()
