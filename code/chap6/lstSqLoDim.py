# Figure 6.22
# Least squares regression lines when there are (i) 20 observations
# and (ii) 2 observations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(911)

df_size = 20
error_sd = 5.0

my_df = pd.DataFrame({'x': np.random.normal(size=df_size)})
my_df['y'] = 7 * my_df['x'] + np.random.normal(scale=error_sd, size=df_size)
small_df = my_df.iloc[:2]

lm_model = smf.ols('y ~ x', data=my_df)
lm_fit = lm_model.fit()
lm_model2 = smf.ols('y ~ x', data=small_df)
lm_fit2 = lm_model2.fit()
my_fits = [lm_fit, lm_fit2]
my_data = [my_df, small_df]
x_array = np.linspace(my_df['x'].min(), my_df['x'].max())

fig = plt.figure(figsize=(8, 4))
for i in range(2):
    ax = fig.add_subplot(1, 2, i+1)
    my_data[i].plot(x='x', y='y', c='r', kind='scatter', alpha=0.7, ax=ax)
    ax.plot(x_array, my_fits[i].predict(
        dict(x=x_array)), c='k', linewidth=1, alpha=0.7)
    ax.set_xlim(my_df['x'].min(), my_df['x'].max())
    ax.set_ylim(my_df['y'].min(), my_df['y'].max())
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')

fig.tight_layout()
