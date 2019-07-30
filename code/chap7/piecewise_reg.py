# Plot figure 7.2
# Piecewise regression of wage versus age
# Confidence intervals are plotted for wage versus age plot only
# When y variable is a dummy, statsmodels does not provide confidence intervals


from statsmodels import datasets
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

wage = datasets.get_rdataset('Wage', 'ISLR').data

wage['age_group'] = pd.cut(wage['age'], bins=[10, 35, 50, 65, 100],
                           labels=['below35', 'f35to50', 'f50to65', 'above65'])

glm_model = smf.glm('wage ~ age_group', data=wage)
glm_fit = glm_model.fit()

res_df = wage[['age', 'wage']].copy()
res_df['wage_fit'] = glm_fit.get_prediction().predicted_mean
res_df['wage_lower'] = glm_fit.get_prediction().conf_int()[:, 0]
res_df['wage_upper'] = glm_fit.get_prediction().conf_int()[:, 1]
res_df.sort_values('age', inplace=True)

fig = plt.figure(figsize=(7, 4))
ax1 = fig.add_subplot(121)
res_df.plot(x='age', y='wage', kind='scatter', s=10, alpha=0.5, ax=ax1)
res_df.plot(x='age', y='wage_fit', c='k', ax=ax1)
res_df.plot(x='age', y='wage_lower', c='r', linestyle='--', ax=ax1)
res_df.plot(x='age', y='wage_upper', c='r', linestyle='-.', ax=ax1)
ax1.set_xlabel('Age')
ax1.set_ylabel('Wage')

wage['wage_gt250'] = wage['wage'].apply(lambda x: 1 if x > 250 else 0)
model_250 = smf.logit('wage_gt250 ~ age_group', data=wage)
fit_250 = model_250.fit()

res_250 = wage[['age', 'wage_gt250']].copy()
res_250['prob_gt250'] = fit_250.predict()
res_250.sort_values('age', inplace=True)

ax2 = fig.add_subplot(122)
res_250.plot(x='age', y='prob_gt250', c='k', ax=ax2)
ax2.scatter(res_250['age'], 0.1 * res_250['wage_gt250'], marker='|',
            color='grey', alpha=0.5)
ax2.set_xlabel('Age')
ax2.set_ylabel('Prob(Wage > 250 | Age)')

fig.suptitle('Piecewise Constant')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
