# Figure 6.23
# Simulated data with upto 20 features, all unrelated to dependent variable y
# As more featues are added to OLS regression, training error decreases,
# but test error increases

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

n_obs = 20
p_features = 20
np.random.seed(911)

X_train = np.random.normal(size=(n_obs, p_features))
X_test = np.random.normal(size=(n_obs, p_features))

y_train = np.random.normal(size=n_obs)
y_test = np.random.normal(size=n_obs)

train_rsq = []
train_mse = []
test_mse = []
num_vars = []

for p in range(1, p_features + 1):
    lm_model = LinearRegression()
    lm_model.fit(X_train[:, :p], y_train)
    y_fit_train = lm_model.predict(X_train[:, :p])
    train_mse.append(np.mean((y_fit_train - y_train) ** 2))
    train_rsq.append(1 - np.sum((y_fit_train - y_train) ** 2) /
                     np.sum((y_train - np.mean(y_train)) ** 2))
    y_fit_test = lm_model.predict(X_test[:, :p])
    test_mse.append(np.mean((y_fit_test - y_test) ** 2))
    num_vars.append(p)

fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(131)
ax1.plot(num_vars, train_rsq)
ax1.set_ylabel(r'$R^2$')

ax2 = fig.add_subplot(132)
ax2.plot(num_vars, train_mse)
ax2.set_ylabel('Training MSE')

ax3 = fig.add_subplot(133)
ax3.plot(num_vars, test_mse)
ax3.set_ylabel('Test MSE')

for ax in fig.axes:
    ax.set_xlabel('Number of Variables')
    ax.set_xticks([5, 10, 15, 20])

fig.tight_layout()
