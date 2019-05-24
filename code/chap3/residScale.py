# Plot figure 3.11

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

np.random.seed(911)
my_df = pd.DataFrame({'x': np.linspace(10.001, 30, 2000)})
my_df['y'] = my_df['x'] + \
    np.random.normal(loc=0.0, scale=0.2 * my_df['x'], size=my_df.shape[0])

reg_model = smf.ols(formula='y ~ x', data=my_df)
reg_fit = reg_model.fit()
resid_lowess = lowess(endog=reg_fit.resid, exog=reg_fit.fittedvalues)
x_bins = np.linspace(10, 30, 21)
x_groups = pd.cut(my_df['x'], x_bins)
resid_quantile_df = pd.DataFrame({'x_mid': my_df['x'].groupby(x_groups).mean()})
resid_quantile_df['resid_05'] = reg_fit.resid.groupby(x_groups).quantile(0.05)
resid_quantile_df['resid_95'] = reg_fit.resid.groupby(x_groups).quantile(0.95)

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)
ax1.scatter(reg_fit.fittedvalues, reg_fit.resid, color='grey', alpha=0.2)
ax1.plot(resid_lowess[:,0], resid_lowess[:,1], c='r')
resid_quantile_df.plot(x='x_mid', y='resid_05', c='b', linestyle='--', ax=ax1)
resid_quantile_df.plot(x='x_mid', y='resid_95', c='b', linestyle='--', ax=ax1)
ax1.set_xlabel('Fitted values')
ax1.set_ylabel('Residuals')
ax1.set_title('Response Y')
ax1.set_xlim(10, 30)
ax1.get_legend().remove()

reg_model_log = smf.ols(formula='np.log(y) ~ x', data=my_df)
reg_fit_log = reg_model_log.fit()
resid_lowess_log = lowess(endog=reg_fit_log.resid,
                          exog=np.exp(reg_fit_log.fittedvalues))
resid_quantile_df = pd.DataFrame({'x_mid': my_df['x'].groupby(x_groups).mean()})
resid_quantile_df['resid_05'] = \
    reg_fit_log.resid.groupby(x_groups).quantile(0.05)
resid_quantile_df['resid_95'] = \
    reg_fit_log.resid.groupby(x_groups).quantile(0.95)
ax2 = fig.add_subplot(122)
ax2.scatter(np.exp(reg_fit_log.fittedvalues), reg_fit_log.resid, color='grey',
            alpha=0.2)
ax2.plot(resid_lowess_log[:,0], resid_lowess_log[:,1], c='r')
resid_quantile_df.plot(x='x_mid', y='resid_05', c='b', linestyle='--', ax=ax2)
resid_quantile_df.plot(x='x_mid', y='resid_95', c='b', linestyle='--', ax=ax2)
ax2.set_xlabel('Fitted values')
ax2.set_ylabel('Residuals')
ax2.set_title('Response log(Y)')
ax2.set_xlim(10, 30)
ax2.get_legend().remove()

fig.tight_layout()
