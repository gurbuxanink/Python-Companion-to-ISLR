# OLS regression plots of Sales versus advertising in TV, radio, newspaper
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

advertising = pd.read_csv('data/Advertising.csv', index_col=0)

fig = plt.figure()
ax1 = fig.add_subplot(131)
tv_model = smf.ols(formula='sales ~ TV', data=advertising)
tv_model_fit = tv_model.fit()
advertising.plot(x='TV', y='sales', kind='scatter', legend=False, alpha=0.7,
                 ax=ax1)
ax1.set_ylabel('Sales')
tv_exog = np.linspace(advertising['TV'].min(), advertising['TV'].max())
ax1.plot(tv_exog, tv_model_fit.predict(exog=dict(TV=tv_exog)), c='r')

ax2 = fig.add_subplot(132)
radio_model = smf.ols(formula='sales ~ radio', data=advertising)
radio_model_fit = radio_model.fit()
advertising.plot(x='radio', y='sales', kind='scatter', legend=False, alpha=0.7,
                 ax=ax2)
ax2.set_xlabel('Radio')
ax2.set_ylabel('')
radio_exog = np.linspace(advertising['radio'].min(),
                         advertising['radio'].max())
ax2.plot(radio_exog, radio_model_fit.predict(exog=dict(radio=radio_exog)), c='r')

ax3 = fig.add_subplot(133)
newspaper_model = smf.ols(formula='sales ~ newspaper', data=advertising)
newspaper_model_fit = newspaper_model.fit()
advertising.plot(x='newspaper', y='sales', kind='scatter', legend=False,
                 alpha=0.7, ax=ax3)
ax3.set_xlabel('Newspaper')
ax3.set_ylabel('')
newspaper_exog = np.linspace(advertising['newspaper'].min(),
                             advertising['newspaper'].max())
ax3.plot(newspaper_exog, newspaper_model_fit.predict(
    exog=dict(newspaper=newspaper_exog)), c='r')

fig.tight_layout()
