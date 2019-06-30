# Figure 6.6
# Plot of of lasso coefficients versus lambda (alpha in sklearn)

from statsmodels import datasets
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

credit = datasets.get_rdataset('Credit', 'ISLR').data
credit.drop(columns='ID')

X_numeric = credit[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']]
X_numeric_std = X_numeric.copy() / X_numeric.std()
X_categorical = credit[['Gender', 'Student', 'Married', 'Ethnicity']]
X_cat = pd.get_dummies(X_categorical)
X = pd.concat((X_numeric_std, X_cat), axis=1)
y = credit['Balance']

alpha_vals = np.logspace(1, 4)
lasso_coef = []
for alpha in alpha_vals:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    lasso_coef.append(lasso.coef_)

my_keys = ['Income', 'Limit', 'Rating', 'Student_No']
key_coef = {}
for key in my_keys:
    key_ind = np.where(X.columns == key)[0][0]
    key_coef[key] = [coef[key_ind] for coef in lasso_coef]
key_linestyles = {'Income': '-', 'Limit': '--', 'Rating': ':',
                  'Student_No': '-.'}
key_colors = {'Income': 'k', 'Limit': 'r', 'Rating': 'g', 'Student_No': 'b'}

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
for key in key_coef.keys():
    ax1.plot(alpha_vals, key_coef[key], c=key_colors[key],
             linestyle=key_linestyles[key], label=key)
ax1.legend()
ax1.set_xlabel(r'$\lambda$')
ax1.set_xscale('log')

ols_model = LinearRegression()
ols_model.fit(X, y)
ols_coef = ols_model.coef_
ols_coef_norm = np.sum(np.abs(ols_coef))

lasso_coef_ratio = []
for coef in lasso_coef:
    lasso_coef_ratio.append(np.sum(np.abs(coef)) / ols_coef_norm)

ax2 = fig.add_subplot(122)
for key in key_coef.keys():
    ax2.plot(lasso_coef_ratio, key_coef[key], c=key_colors[key],
             linestyle=key_linestyles[key], label=key)
ax2.legend()
ax2.set_xlabel(r'$\|| \hat{\beta}_\lambda^L \||_1 / \|| \hat{\beta} \||_1$')

for ax in fig.axes:
    ax.set_ylabel('Standardized Coefficients')
fig.tight_layout()
