# Figure 7.1
# Polynomial fit along with confidence intervals for Wage data
# Confidence intervals are drawn for ols regression only
# statsmodels does not provide confidence intervals for logistic regression

from statsmodels import datasets
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

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

# Estimate confidence intervals from the formula
poly = PolynomialFeatures(degree=4)
X_mat = poly.fit_transform(res250_df['age'][:, np.newaxis])
cov_beta = fit_250.cov_params()
predict_var = np.diag(np.dot(X_mat, np.dot(cov_beta, X_mat.T)))
predict_error = np.sqrt(predict_var)
Xb = np.dot(X_mat, fit_250.params)

predict_upper = Xb + 1.96 * predict_error
predict_lower = Xb - 1.96 * predict_error
res250_df['prob_lower'] = np.exp(predict_lower) / (1 + np.exp(predict_lower))
res250_df['prob_upper'] = np.exp(predict_upper) / (1 + np.exp(predict_upper))

ax2 = fig.add_subplot(122)
ax2.scatter(wage['age'], 0.5 * wage['wage_gt250'], marker='|', alpha=0.5,
            color='grey')
res250_df.plot(x='age', y='prob_gt250', c='k', ax=ax2)
res250_df.plot(x='age', y='prob_lower', c='r', linestyle='--', ax=ax2)
res250_df.plot(x='age', y='prob_upper', c='r', linestyle='-.', ax=ax2)
ax2.set_yticks([0, 0.2, 0.4, 0.6])
ax2.set_ylabel('Prob(Wage > 250 | Age)')
ax2.set_xlabel('Age')

fig.suptitle('Degree-4 Polynomial')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
