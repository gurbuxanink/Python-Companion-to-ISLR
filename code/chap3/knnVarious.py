# Plot figure 3.19

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.formula.api as smf

def trueFunc(x):
    return 2 + 3 * x - x ** 3

error_sd = 0.15
x_min = -1.0
x_max = 1.0
df_size = 50

np.random.seed(911)
train_df = pd.DataFrame({'x': np.random.uniform(x_min, x_max, df_size)})
train_df['y'] = np.vectorize(trueFunc)(train_df['x']) + \
    np.random.normal(loc=0.0, scale=error_sd, size=train_df.shape[0])

test_df = pd.DataFrame({'x': np.random.uniform(x_min, x_max, df_size)})
test_df['y'] = np.vectorize(trueFunc)(test_df['x']) + \
    np.random.normal(loc=0.0, scale=error_sd, size=test_df.shape[0])

ols_reg = smf.ols(formula='y ~ x', data=train_df)
ols_fit = ols_reg.fit()
y_pred_ols = ols_fit.predict(exog=dict(x=test_df['x']))
mse_ols = np.mean((y_pred_ols - test_df['y']) ** 2)

knn1_reg = KNeighborsRegressor(n_neighbors=1)
knn1_reg.fit(train_df['x'].values.reshape(-1, 1), train_df['y'])

knn9_reg = KNeighborsRegressor(n_neighbors=9)
knn9_reg.fit(train_df['x'].values.reshape(-1, 1), train_df['y'])

x_array = np.linspace(x_min, x_max, 400)
y_array_true = np.vectorize(trueFunc)(x_array)
y_array1 = knn1_reg.predict(x_array.reshape(-1, 1))
y_array9 = knn9_reg.predict(x_array.reshape(-1, 1))

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(221)
ax1.plot(x_array, y_array_true, c='k', alpha=0.7)
ax1.plot(x_array, y_array1, c='b', linestyle='-.', alpha=0.7)
ax1.plot(x_array, y_array9, c='r', linestyle='--', alpha=0.7)
ax1.set_xlabel('x')
ax1.set_ylabel('y')


k_vals = []
knn_mse = []
for k in range(1, 21):
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(train_df['x'].values.reshape(-1, 1), train_df['y'])
    y_knn = knn_reg.predict(test_df['x'].values.reshape(-1, 1))
    mse = np.mean((y_knn - test_df['y']) ** 2)
    k_vals.append(k)
    knn_mse.append(mse)

k_inv = [1/k for k in k_vals]
ax2 = fig.add_subplot(222)
ax2.axhline(y=mse_ols, linestyle='--', color='grey')
ax2.plot(k_inv, knn_mse, linestyle='-.', c='g')
ax2.set_xscale('log')
ax2.set_xlabel(r'$\frac{1}{K}$')
ax2.set_ylabel('Mean Squared Error')
ax2.set_xticks([0.1, 0.5, 1.0])
ax2.set_xticklabels([0.1, 0.5, 1.0])

def myFunc(x):
    return 2 + 3 * x - 2.5 * x ** 3

np.random.seed(911)
train_df = pd.DataFrame({'x': np.random.uniform(x_min, x_max, df_size)})
train_df['y'] = np.vectorize(myFunc)(train_df['x']) + \
    np.random.normal(loc=0.0, scale=error_sd, size=train_df.shape[0])

test_df = pd.DataFrame({'x': np.random.uniform(x_min, x_max, df_size)})
test_df['y'] = np.vectorize(myFunc)(test_df['x']) + \
    np.random.normal(loc=0.0, scale=error_sd, size=test_df.shape[0])

ols_reg = smf.ols(formula='y ~ x', data=train_df)
ols_fit = ols_reg.fit()
y_pred_ols = ols_fit.predict(exog=dict(x=test_df['x']))
mse_ols = np.mean((y_pred_ols - test_df['y']) ** 2)

knn1_reg = KNeighborsRegressor(n_neighbors=1)
knn1_reg.fit(train_df['x'].values.reshape(-1, 1), train_df['y'])

knn9_reg = KNeighborsRegressor(n_neighbors=9)
knn9_reg.fit(train_df['x'].values.reshape(-1, 1), train_df['y'])

x_array = np.linspace(x_min, x_max, 400)
y_array_true = np.vectorize(myFunc)(x_array)
y_array1 = knn1_reg.predict(x_array.reshape(-1, 1))
y_array9 = knn9_reg.predict(x_array.reshape(-1, 1))

ax3 = fig.add_subplot(223)
ax3.plot(x_array, y_array_true, c='k', alpha=0.7)
ax3.plot(x_array, y_array1, c='b', linestyle='-.', alpha=0.7)
ax3.plot(x_array, y_array9, c='r', linestyle='--', alpha=0.7)
ax3.set_xlabel('x')
ax3.set_ylabel('y')


k_vals = []
knn_mse = []
for k in range(1, 21):
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(train_df['x'].values.reshape(-1, 1), train_df['y'])
    y_knn = knn_reg.predict(test_df['x'].values.reshape(-1, 1))
    mse = np.mean((y_knn - test_df['y']) ** 2)
    k_vals.append(k)
    knn_mse.append(mse)

k_inv = [1/k for k in k_vals]
ax4 = fig.add_subplot(224)
ax4.axhline(y=mse_ols, linestyle='--', color='grey')
ax4.plot(k_inv, knn_mse, linestyle='-.', c='g')
ax4.set_xscale('log')
ax4.set_xlabel(r'$\frac{1}{K}$')
ax4.set_ylabel('Mean Squared Error')
ax4.set_xticks([0.1, 0.5, 1.0])
ax4.set_xticklabels([0.1, 0.5, 1.0])

fig.tight_layout()
