# Figure 7.1
# Polynomial fit along with confidence intervals for Wage data
# Confidence intervals are drawn for ols regression only
# statsmodels does not provide confidence intervals for logistic regression

from statsmodels import datasets
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

wage = datasets.get_rdataset('Wage', 'ISLR').data

wage_model = smf.ols('wage ~ age + I(age ** 2) + I(age ** 3) + I(age ** 4)',
                     data=wage)
wage_fit = wage_model.fit()

res_df = pd.DataFrame(
    {'age': wage['age'],
     'wage_fit': wage_fit.fittedvalues,
     'wage_lower': wage_fit.get_prediction().conf_int()[:, 0],
     'wage_upper': wage_fit.get_prediction().conf_int()[:, 1]})
res_df.sort_values('age', inplace=True)

fig = plt.figure(figsize=(7, 4))
ax1 = fig.add_subplot(121)
wage.plot(x='age', y='wage', kind='scatter', s=10, alpha=0.5, ax=ax1)
res_df.plot(x='age', y='wage_fit', c='k', ax=ax1)
res_df.plot(x='age', y='wage_lower', linestyle='--', c='r', ax=ax1)
res_df.plot(x='age', y='wage_upper', linestyle='-.', c='r', ax=ax1)
ax1.set_ylabel('Wage')
ax1.set_xlabel('Age')

wage['wage_gt250'] = wage['wage'].apply(lambda x: 1 if x > 250 else 0)

model_250 = smf.logit(
    'wage_gt250 ~ age + I(age ** 2) + I(age ** 3) + I(age ** 4)', data=wage)
fit_250 = model_250.fit()
res250_df = pd.DataFrame({'age': np.linspace(wage['age'].min(),
                                             wage['age'].max())})
res250_df['prob_gt250'] = fit_250.predict(res250_df)

ax2 = fig.add_subplot(122)
res250_df.plot(x='age', y='prob_gt250', ax=ax2)
ax2.scatter(wage['age'], 0.2 * wage['wage_gt250'], marker='|', alpha=0.5,
            color='grey')
ax2.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
ax2.set_ylabel('Prob(Wage > 250 | Age)')
ax2.set_xlabel('Age')

fig.suptitle('Degree-4 Polynomial')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
