import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

auto_full = pd.read_csv('data/Auto.csv', na_values='?')
auto = auto_full.dropna()

reg1_model = smf.ols(formula='mpg ~ horsepower', data=auto)
reg1_fit = reg1_model.fit()

reg2_model = smf.ols(formula='mpg ~ horsepower + I(horsepower**2)', data=auto)
reg2_fit = reg2_model.fit()

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)

resid_df = pd.DataFrame(dict(fit_val=reg1_fit.fittedvalues,
			     resid=reg1_fit.resid))
resid_lowess = lowess(resid_df['resid'], resid_df['fit_val'])

ax1.scatter(reg1_fit.fittedvalues, reg1_fit.resid, color='grey', alpha=0.6)
ax1.axhline(y=0, color='grey', alpha=0.6)
ax1.plot(resid_lowess[:,0], resid_lowess[:,1], c='r')
ax1.set_xlabel('Fitted values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residual Plot for Linear Fit')

ax2 = fig.add_subplot(122)
resid_df = pd.DataFrame(dict(fit_val=reg2_fit.fittedvalues,
			     resid=reg2_fit.resid))
resid_lowess = lowess(resid_df['resid'], resid_df['fit_val'])
ax2.scatter(reg2_fit.fittedvalues, reg2_fit.resid, color='grey', alpha=0.6)
ax2.axhline(y=0, color='grey', alpha=0.6)
ax2.plot(resid_lowess[:,0], resid_lowess[:,1], c='r')
ax2.set_xlabel('Fitted values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residual Plot for Quadratic Fit')

fig.tight_layout()
