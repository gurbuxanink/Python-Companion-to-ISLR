import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

advertising = pd.read_csv('data/Advertising.csv', index_col=0)
advert_model = smf.ols(formula='sales ~ TV', data=advertising)
advert_fit = advert_model.fit()

x_tv = np.linspace(advertising['TV'].min() - 5, advertising['TV'].max() + 5)
y_sales_pred = advert_fit.predict(exog=dict(TV=x_tv))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x_tv, y_sales_pred, alpha=0.7, c='b')
ax1.vlines(x=advertising['TV'], ymin=advert_fit.fittedvalues,
           ymax=advertising['sales'], color='grey', alpha=0.7, linewidth=0.7)
advertising.plot(x='TV', y='sales', kind='scatter', c='r', alpha=0.7, ax=ax1,
                 s=10)
ax1.set_ylabel('Sales')
