# Figure 6.7
# Contours of error and constraint functions for lasso and ridge regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

cor = 0.3
error_sd = 1.0
p_features = 2
df_size = 100
np.random.seed(911)

cov_mat = np.diag(np.repeat(1 - cor, p_features))
cov_mat = cov_mat + cor
X = np.random.multivariate_normal(mean=np.zeros((p_features,)),
                                  cov=cov_mat, size=df_size)
true_beta = np.array([1, 1.5])
y = np.dot(X, true_beta) + np.random.normal(scale=error_sd, size=df_size)

ols_model = LinearRegression()
ols_model.fit(X, y)
beta = ols_model.coef_
alpha = ols_model.intercept_

beta1_x = np.linspace(-1 * beta[0], 3 * beta[0], num=200)
beta2_x = np.linspace(-1 * beta[1], 3 * beta[1], num=200)
beta1_grid, beta2_grid = np.meshgrid(beta1_x, beta2_x)


def rssFunc(alpha, beta, X, y):
    y_fit = np.dot(X, beta) + alpha
    error = y_fit - y
    return np.sum(error ** 2)


beta_df = pd.DataFrame({'beta1': beta1_grid.ravel(),
                        'beta2': beta2_grid.ravel()})
rss_array = beta_df.apply(lambda row: rssFunc(alpha, row, X, y), axis=1)
rss_grid = np.reshape(np.array(rss_array), beta1_grid.shape)

s = 0.8
beta_df['lasso_include'] = beta_df.apply(lambda row:
                                         np.sum(np.abs(row[:2])) <= s, axis=1)
beta_df['ridge_include'] = beta_df.apply(lambda row:
                                         np.sum(row[:2] ** 2) <= s, axis=1)

rss_levels = [rssFunc(alpha, [0, s], X, y), rssFunc(alpha, [s, 0], X, y),
              rssFunc(alpha, 1.2 * beta, X, y),
              rssFunc(alpha, 1.5 * beta, X, y)]
rss_levels.sort()

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

beta_df.loc[beta_df['lasso_include']].plot(x='beta1', y='beta2',
                                           marker='.', color='cyan', alpha=0.2,
                                           kind='scatter', ax=ax1)
beta_df.loc[beta_df['ridge_include']].plot(x='beta1', y='beta2', marker='.',
                                           color='cyan', alpha=0.2,
                                           kind='scatter', ax=ax2)

for ax in fig.axes:
    ax.contour(beta1_grid, beta2_grid, rss_grid, levels=rss_levels,
               colors='red', alpha=0.7)
    ax.axhline(y=0, color='grey', alpha=0.7)
    ax.axvline(x=0, color='grey', alpha=0.7)
    ax.scatter(beta[0], beta[1], c='k')
    ax.text(0.8 * beta[0], beta[1], r'$\hat{\beta}$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')

fig.tight_layout()
