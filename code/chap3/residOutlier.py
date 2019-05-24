# Plot figure 3.12

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(911)
my_df = pd.DataFrame({'x': np.linspace(-2, 2, 41)})
my_df['y'] = 2 * my_df['x'] + 2 + \
    np.random.normal(loc=0, scale=0.5, size=my_df.shape[0])
my_df.loc[20, 'y'] = my_df.loc[20, 'y'] + 5

fig = plt.figure(figsize=(9,3))

ax1 = fig.add_subplot(131)
my_df.plot(x='x', y='y', kind='scatter', color='grey', alpha=0.6, ax=ax1)
ax1.scatter(my_df.iloc[20]['x'], my_df.iloc[20]['y'], c='r')
all_reg = smf.ols(formula='y ~ x', data=my_df)
all_fit = all_reg.fit()
wo_outlier_reg = smf.ols(formula='y ~ x', data=my_df.drop(labels=[20]))
wo_outlier_fit = wo_outlier_reg.fit()
ax1.plot(my_df['x'], all_fit.fittedvalues, c='r')
ax1.plot(my_df.drop(labels=[20])['x'], wo_outlier_fit.fittedvalues, c='b',
         linestyle='--')
ax1.text(my_df.iloc[20]['x'] - 0.5, my_df.iloc[20]['y'] - 0.2, '20', color='red')
ax1.set_xlabel(r'$X$')
ax1.set_ylabel(r'$Y$')

ax2 = fig.add_subplot(132)
ax2.scatter(all_fit.fittedvalues, all_fit.resid, color='grey', alpha=0.6)
ax2.axhline(y=0, color='grey')
ax2.scatter(all_fit.fittedvalues[20], all_fit.resid[20], c='r')
ax2.text(all_fit.fittedvalues[20] - 1, all_fit.resid[20] - 0.2, '20',
         color='red')
ax2.set_xlabel('Fitted values')
ax2.set_ylabel('Residuals')

all_infl = all_fit.get_influence()
ax3 = fig.add_subplot(133)
ax3.scatter(all_fit.fittedvalues, all_infl.resid_studentized_internal,
            color='grey', alpha=0.6)
ax3.axhline(y=0, color='grey')
ax3.scatter(all_fit.fittedvalues[20], all_infl.resid_studentized_internal[20],
            c='r')
ax3.text(all_fit.fittedvalues[20] - 1,
         all_infl.resid_studentized_internal[20] - 0.2, '20', color='red')
ax3.set_xlabel('Fitted values')
ax3.set_ylabel('Studentized residuals')

fig.tight_layout()
