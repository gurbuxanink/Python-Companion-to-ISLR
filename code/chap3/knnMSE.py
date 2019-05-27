# Plot figure 3.18
# Generate train and test data using given function and sd_error
# Fit KNN regression for a range of n_neighbors, calculate test MSE
# Fite ols regression, calculate test MSE
# Plot regression line, MSE versus 1/K

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsRegressor

def trueFunc(x):
    return 2 * x + 2
x_min, x_max = -1.0, 1.0
error_sd = 0.3
k_range=range(1, 21, 2)
train_size=50
test_size=50

np.random.seed(911)
my_df = pd.DataFrame({'x': np.random.uniform(x_min, x_max, size=50)})
my_df['y'] = np.vectorize(trueFunc)(my_df['x']) + \
    np.random.normal(loc=0.0, scale=error_sd, size=my_df.shape[0])

test_df = pd.DataFrame({'x': np.random.uniform(x_min, x_max, size=50)})
test_df['y'] = np.vectorize(trueFunc)(test_df['x']) + \
    np.random.normal(0, scale=error_sd, size=test_df.shape[0])

k_vals = []
k_mse = []
for k in k_range:
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(my_df['x'].values.reshape(-1, 1), my_df['y'])

    y_pred_knn = knn_reg.predict(test_df['x'].values.reshape(-1, 1))
    mse = np.mean((y_pred_knn - test_df['y']) ** 2)
    # y_pred_knn = knn_reg.predict(my_df['x'].values.reshape(-1, 1))
    # mse = np.mean((y_pred_knn - my_df['y']) ** 2)
    k_vals.append(k)
    k_mse.append(mse)

ols_reg = smf.ols(formula='y ~ x', data=my_df)
ols_fit = ols_reg.fit()
y_pred_ols = ols_fit.predict(exog=dict(x=test_df['x']))
ols_mse = np.mean((y_pred_ols - test_df['y']) ** 2)

x_array = np.linspace(x_min, x_max)
y_true = np.vectorize(trueFunc)(x_array)
y_fit_ols = ols_fit.predict(exog=dict(x=x_array))

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
my_df.plot(x='x', y='y', kind='scatter', c='r', s=7, alpha=0.7, ax = ax1)
ax1.plot(x_array, y_true, c='k', alpha=0.7)
ax1.plot(x_array, y_fit_ols, c='b', linestyle='--', alpha=0.7)

inv_k = [1/k for k in k_vals]
ax2 = fig.add_subplot(122)
ax2.axhline(y=ols_mse, linestyle='--', color='grey')
ax2.plot(inv_k, k_mse, c='g', linestyle='-.')
# ax2.set_xscale('log')
ax2.set_xlabel(r'$\frac{1}{K}$')
ax2.set_ylabel('Mean Squared Error')

fig.tight_layout()
