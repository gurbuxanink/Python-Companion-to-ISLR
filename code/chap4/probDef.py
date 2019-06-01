# Plot figure 4.2

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels import datasets
import pandas as pd
import numpy as np

default = datasets.get_rdataset('Default', 'ISLR').data

default['default_cat'] = default.apply(lambda x: int(x['default'] == 'Yes'),
                                       axis=1)

linear_model = smf.ols(formula='default_cat ~ balance', data=default)
linear_fit = linear_model.fit()

balance_xx = np.linspace(default['balance'].min(), default['balance'].max())

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
default.plot(x='balance', y='default_cat', kind='scatter', alpha=0.5, ax=ax1,
             color='brown')
ax1.plot(balance_xx, linear_fit.predict(exog=dict(balance=balance_xx)), c='b',
         linestyle='--')
ax1.axhline(y=0, linestyle='--', color='grey')
ax1.axhline(y=1, linestyle='--', color='grey')
ax1.set_xlabel('Balance')
ax1.set_ylabel('Probability of Default')

logit_model = smf.logit(formula='default_cat ~ balance', data=default)
logit_fit = logit_model.fit()

ax2 = fig.add_subplot(122)
default.plot(x='balance', y='default_cat', kind='scatter', alpha=0.5, ax=ax2,
             color='brown')
ax2.plot(balance_xx, logit_fit.predict(exog=dict(balance=balance_xx)), c='b',
         linestyle='--')
ax2.axhline(y=0, linestyle='--', color='grey')
ax2.axhline(y=1, linestyle='--', color='grey')
ax2.set_xlabel('Balance')
ax2.set_ylabel('Probability of Default')

fig.tight_layout()
