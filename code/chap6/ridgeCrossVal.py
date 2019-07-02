# Figure 6.12
# Use leave-one-out cross-validation error to select best labmda parameter
# for ridge regression

from statsmodels import datasets
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

credit = datasets.get_rdataset('Credit', 'ISLR').data
credit.drop(columns='ID', inplace=True)
X_numeric = credit[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']]
X_numeric_std = X_numeric / X_numeric.std()
X_categoric = credit[['Gender', 'Student', 'Married', 'Ethnicity']]
X_cat = pd.get_dummies(X_categoric)
X = pd.concat((X_numeric_std, X_cat), axis=1)
y = credit['Balance']

index = np.arange(X.shape[0])
alpha_vals = np.logspace(-3, 1, 40)
cv_error = []
ridge_coef = []

for alpha in alpha_vals:
    ridge_model = Ridge(alpha=alpha)
    error = []
    for i in index:
        X_train = X.loc[index != i]
        y_train = y.loc[index != i]
        X_test = X.loc[index == i]
        y_test = y.loc[index == i]
        ridge_model.fit(X_train, y_train)
        y_predict = ridge_model.predict(X_test)
        error.append((y_predict[0] - y_test[i]) ** 2)
    cv_error.append(np.sqrt(np.mean(error)))
    ridge_model.fit(X, y)
    ridge_coef.append(ridge_model.coef_)

x_vars = ['Income', 'Limit', 'Rating', 'Student_Yes']
x_coef = {}
for x in x_vars:
    x_ind = np.where(X.columns == x)[0][0]
    x_coef[x] = [coef[x_ind] for coef in ridge_coef]

x_linestyles = {'Income': '-', 'Limit': '--', 'Rating': ':',
                'Student_Yes': '-.'}
x_colors = {'Income': 'k', 'Limit': 'r', 'Rating': 'g', 'Student_Yes': 'b'}

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.plot(alpha_vals, cv_error)
ax1.set_ylabel('Cross-Validation Error')

ax2 = fig.add_subplot(122)
for x_name in x_vars:
    ax2.plot(alpha_vals, x_coef[x_name], c=x_colors[x_name],
             linestyle=x_linestyles[x_name], label=x_name)
ax2.legend(loc=(0.05, 0.2))
ax2.set_ylabel('Standardized Coefficients')

min_ind = np.argmin(cv_error)

for ax in fig.axes:
    ax.set_xscale('log')
    ax.axvline(x=alpha_vals[min_ind], linestyle='--', color='grey')
    ax.set_xlabel(r'$\lambda$')

fig.tight_layout()
