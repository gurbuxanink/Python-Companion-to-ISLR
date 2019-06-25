# Figure 6.4
# Plot standardized coefficients as function of lambda (alpha in sklearn)
# and ratio of l-2 norm of ridge regression coefficients and l-2 norm
# of OLS regression coefficients

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from statsmodels import datasets
import pandas as pd

credit = datasets.get_rdataset('Credit', 'ISLR').data
credit.drop(columns='ID', inplace=True)

X_numeric = credit[['Income', 'Limit', 'Rating',
                    'Cards', 'Age', 'Education']].copy()
X_num_std = (X_numeric - X_numeric.mean()) / X_numeric.std()
# X_num_std = X_numeric / X_numeric.std()
X_cat = credit[['Gender', 'Student', 'Married', 'Ethnicity']].copy()
X_cat = pd.get_dummies(X_cat)
X = pd.concat((X_num_std, X_cat), axis=1)
y = credit['Balance']
alpha_vals = np.logspace(-3, 4, 49)
beta_ridge = []
intercept_ridge = []
for alpha in alpha_vals:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X, y)
    beta_ridge.append(ridge_model.coef_)
    intercept_ridge.append(ridge_model.intercept_)

income_coef = [beta_ridge[i][0] for i in range(len(alpha_vals))]
limit_coef = [beta_ridge[i][1] for i in range(len(alpha_vals))]
rating_coef = [beta_ridge[i][2] for i in range(len(alpha_vals))]
student_coef = [beta_ridge[i][9] for i in range(len(alpha_vals))]

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.plot(alpha_vals, income_coef, label='Income', c='k')
ax1.plot(alpha_vals, limit_coef, label='Limit', c='r', linestyle='--')
ax1.plot(alpha_vals, rating_coef, label='Rating', c='b', linestyle=':')
ax1.plot(alpha_vals, student_coef, label='Student', color='brown',
         linestyle='-.')
ax1.legend(loc='lower left', frameon=False)
ax1.set_xscale('log')
ax1.set_xlabel(r'$\lambda$')


ols_model = LinearRegression()
ols_model.fit(X, y)
beta_l2_ols = np.sqrt(np.sum([beta ** 2 for beta in ols_model.coef_]))

beta_l2_ratio = []
for i in range(len(beta_ridge)):
    beta_l2_ridge = np.sqrt(np.sum(beta_ridge[i] ** 2))
    beta_l2_ratio.append(beta_l2_ridge / beta_l2_ols)

ax2 = fig.add_subplot(122)
ax2.plot(beta_l2_ratio, income_coef, c='k')
ax2.plot(beta_l2_ratio, limit_coef, c='r', linestyle='--')
ax2.plot(beta_l2_ratio, rating_coef, c='b', linestyle=':')
ax2.plot(beta_l2_ratio, student_coef, color='brown', linestyle='-.')
# ax2.set_xscale('log')
ax2.set_xlabel(r'$\|| \hat{\beta}_\lambda^R \||_2 / \|| \hat{\beta} \||_2$')

for ax in fig.axes:
    ax.set_ylabel('Standardized Coefficients')

fig.tight_layout()
