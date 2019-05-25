# Plot figure 3.13

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

np.random.seed(911)
my_df = pd.DataFrame({'x': np.linspace(-2, 2, 41)})
my_df['y'] = 2 + 2 * my_df['x'] + \
    np.random.normal(loc=0, scale=0.5, size=my_df.shape[0])
my_df.loc[20, 'y'] = my_df.loc[20, 'y'] + 5
my_df = my_df.append(pd.DataFrame(dict(x=[4], y=[15])), ignore_index=True)

all_reg = smf.ols(formula='y ~ x', data=my_df)
all_fit = all_reg.fit()
wo_outlier_reg = smf.ols(formula='y ~ x', data=my_df.drop(labels=[41]))
wo_outlier_fit = wo_outlier_reg.fit()

fig = plt.figure(figsize=(9, 3))

ax1 = fig.add_subplot(131)
my_df.plot(x='x', y='y', kind='scatter', color='grey', alpha=0.7, ax=ax1)
ax1.scatter(my_df.iloc[[20, 41]]['x'], my_df.iloc[[20, 41]]['y'],
            color='red')
ax1.text(my_df.iloc[20]['x'] - 1, my_df.iloc[20]['y'] - 0.5, '20',
         color='red')
ax1.text(my_df.iloc[41]['x'] - 1, my_df.iloc[41]['y'] - 0.5, '41',
         color='red')
ax1.plot(my_df['x'], all_fit.fittedvalues, c='r', alpha=0.7)
ax1.plot(my_df['x'], wo_outlier_fit.predict(exog=dict(x=my_df['x'])), c='b',
         linestyle='--', alpha=0.7)
ax1.set_xlabel(r'$X$')
ax1.set_ylabel(r'$Y$')

new_df = pd.DataFrame({'x1': np.linspace(-2, 2, 41)})
new_df['x2'] = new_df['x1'] + np.random.normal(loc=0, scale=0.4,
                                              size=new_df.shape[0])
new_df.loc[30, 'x2'] = new_df.loc[30, 'x2'] - 2
new_df['y'] = new_df['x1'] + new_df['x2'] + \
    np.random.normal(size=new_df.shape[0])

ax2 = fig.add_subplot(132)
new_df.plot(x= 'x1', y='x2', kind='scatter', color='grey', alpha=0.7, ax=ax2)
ax2.plot([-2, 2], [-1.4, 2.6], c='b', linestyle='--', alpha=0.6)
ax2.plot([-2, 2], [-2.6, 1.4], c='b', linestyle='--', alpha=0.6)
ax2.scatter(new_df.loc[30,'x1'], new_df.loc[30, 'x2'], c='r')
ax2.set_xlabel(r'$X_1$')
ax2.set_ylabel(r'$X_2$')

# Verify that observation 30 indeed has high leverage
# new_reg = smf.ols(formula='y ~ x1 + x2', data=new_df)
# new_fit = new_reg.fit()
# sm.graphics.influence_plot(new_fit)

ax3 = fig.add_subplot(133)
sm.graphics.influence_plot(all_fit, ax=ax3)
ax3.set_xlabel(ax3.get_xlabel(), fontsize=10)
ax3.set_ylabel(ax3.get_ylabel(), fontsize=10)
ax3.set_title('')
ax3.set_xlim(0.0, 0.28)
ax3.set_ylim(-3, 8)
ax3.axhline(y=0, color='grey')

fig.tight_layout()
