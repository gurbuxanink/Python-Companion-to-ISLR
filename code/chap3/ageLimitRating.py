# Plot figure 3.15

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.datasets as datasets
import statsmodels.formula.api as smf
import numpy as np

credit = datasets.get_rdataset('Credit', 'ISLR').data

def calcRSS(beta1, beta2, var1, var2, yvar, df):
    # Solve for intercept
    alpha = np.mean(df[yvar] - beta1 * df[var1] - beta2 * df[var2])
    rss = np.sum((df[yvar] - alpha - beta1 * df[var1] - beta2 * df[var2])**2)
    return rss

age_limit_reg = smf.ols(formula='Balance ~ Limit + Age', data=credit)
age_limit_fit = age_limit_reg.fit()
age_limit_res = age_limit_fit.summary2().tables[1]
beta_limit = age_limit_fit.params['Limit']
beta_limit_se = age_limit_res.loc['Limit', 'Std.Err.']
beta_age = age_limit_fit.params['Age']
beta_age_se = age_limit_res.loc['Age', 'Std.Err.']


beta_limit_xx = np.linspace(beta_limit - 4 * beta_limit_se,
                            beta_limit + 4 * beta_limit_se)
beta_age_xx = np.linspace(beta_age - 4 * beta_age_se,
                          beta_age + 4 * beta_age_se)
beta_limit_grid, beta_age_grid = np.meshgrid(beta_limit_xx, beta_age_xx)

rss_df = pd.DataFrame({'beta_limit': beta_limit_grid.ravel(),
                       'beta_age': beta_age_grid.ravel()})
rss_df['rss'] = rss_df.apply(lambda x: calcRSS(x['beta_limit'],
                                               x['beta_age'],
                                               'Limit', 'Age',
                                               'Balance', credit), axis=1)
rss_grid = np.reshape(rss_df['rss'].values, beta_limit_grid.shape)
    
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
cs = ax1.contour(beta_limit_grid, beta_age_grid, rss_grid * 1e-6, colors='b',
                 levels=[21.25, 21.5, 21.8])
ax1.scatter(beta_limit, beta_age, s = 100, c='k')
plt.clabel(cs, inline=1)
ax1.plot([0.15, beta_limit], [beta_age, beta_age], linestyle='--',
         color='grey')
ax1.plot([beta_limit, beta_limit], [-5, beta_age], linestyle='--',
         color='grey')
ax1.set_xlim(beta_limit_grid.min(), beta_limit_grid.max())
ax1.set_ylim(beta_age_grid.min(), beta_age_grid.max())
ax1.set_xlabel(r'$\beta_{Limit}$')
ax1.set_ylabel(r'$\beta_{Age}$')

rating_limit_reg = smf.ols(formula='Balance ~ Rating + Limit', data=credit)
rating_limit_fit = rating_limit_reg.fit()
rating_limit_res = rating_limit_fit.summary2().tables[1]
beta_rating  = rating_limit_fit.params['Rating']
beta_rating_se = rating_limit_res.loc['Rating', 'Std.Err.']
beta_limit = rating_limit_fit.params['Limit']
beta_limit_se = rating_limit_res.loc['Limit', 'Std.Err.']


beta_rating_xx = np.linspace(beta_rating - 4 * beta_rating_se,
                             beta_rating + 4 * beta_rating_se)
beta_limit_xx = np.linspace(beta_limit - 4 * beta_limit_se,
                            beta_limit + 4 * beta_limit_se)
beta_limit_grid, beta_rating_grid = np.meshgrid(beta_limit_xx,
                                                beta_rating_xx)
rss_df = pd.DataFrame({'beta_limit': beta_limit_grid.ravel(),
                       'beta_rating': beta_rating_grid.ravel()})
rss_df['rss'] = rss_df.apply(lambda x: calcRSS(x['beta_limit'],
                                               x['beta_rating'],
                                               'Limit', 'Rating',
                                               'Balance', credit), axis=1)
rss_grid = np.reshape(rss_df['rss'].values, beta_limit_grid.shape)


ax2 = fig.add_subplot(122)
cs = ax2.contour(beta_limit_grid, beta_rating_grid, rss_grid * 1e-6,
                 levels = [21.8, 23], colors='b')
ax2.scatter(beta_limit, beta_rating, s=100, c='k')
plt.clabel(cs, inline=1)
ax2.plot([-0.25, beta_limit], [beta_rating, beta_rating], linestyle='--',
         color='grey')
ax2.plot([beta_limit, beta_limit], [-1.8, beta_rating], linestyle='--',
         color='grey')
ax2.set_xlabel(r'$\beta_{Limit}$')
ax2.set_ylabel(r'$\beta_{Rating}$')
ax2.set_xlim(beta_limit_grid.min(), beta_limit_grid.max())
ax2.set_ylim(beta_rating_grid.min(), beta_rating_grid.max())
fig.tight_layout()
