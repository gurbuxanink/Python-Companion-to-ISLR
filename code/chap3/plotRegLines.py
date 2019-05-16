import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(911)
sd_error = 4.0

my_df = pd.DataFrame({'x': np.linspace(-2, 2, num=100)})
my_df['y_true'] = 2 + 3 * my_df['x']
my_df['y_observe'] = 2 + 3 * my_df['x'] + \
    np.random.normal(loc=0, scale=sd_error, size=my_df.shape[0])

reg_model = smf.ols(formula='y_observe ~ x', data=my_df)
reg_fit = reg_model.fit()
my_df['y_fit'] = reg_fit.fittedvalues

fig = plt.figure()
ax1 = fig.add_subplot(121)
my_df.plot(x='x', y='y_observe', kind='scatter', color='grey', alpha=0.7,
           legend=False, ax=ax1)
my_df.plot(x='x', y='y_true', c='r', legend=False, ax=ax1)
my_df.plot(x='x', y='y_fit', c='b', linestyle='--', legend=False, ax=ax1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_ylim(-12, 12)

ax2 = fig.add_subplot(122)
my_df.plot(x='x', y='y_true', c='r', legend=False, ax=ax2)
my_df.plot(x='x', y='y_fit', c='b', linestyle='--', legend=False, ax=ax2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_ylim(-12, 12)

for i in np.arange(5):
    x = my_df['x']
    y = 2 + 3 * x + np.random.normal(loc=0, scale=sd_error, size=x.size)
    reg_model = smf.ols(formula='y ~ x', data=pd.DataFrame({'x': x, 'y': y}))
    y_fit = reg_model.fit().fittedvalues
    ax2.plot(x, y_fit, c='c', linestyle=':', alpha=0.7)

fig.tight_layout()
