import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# wages = pd.read_csv('data/Wage.csv', index_col=0)
wages = sm.datasets.get_rdataset('Wage', 'ISLR').data
ed_level = wages[['education']].applymap(lambda x: x[:1])
ed_level.rename(index=str, columns={'education': 'ed_level'}, inplace=True)
ed_level.index = np.vectorize(int)(ed_level.index)
wages = wages.merge(ed_level, left_index=True, right_index=True)
wages['const'] = 1
wages['age_2'] = np.vectorize(lambda x: x**2)(wages['age'])
# wages['age_3'] = np.vectorize(lambda x: x**3)(wages['age'])

fig = plt.figure()
ax1 = fig.add_subplot(131)
wages.plot(x='age', y='wage', kind='scatter', ax=ax1, alpha=0.5)
ax1.set_xlabel('Age')
ax1.set_ylabel('Wage')

age_wage_reg = ols(formula='wage ~ age + age_2', data=wages)
age_wage_fit = age_wage_reg.fit()
ax1.scatter(x=np.array(wages['age']), y=age_wage_fit.predict(), marker='+',
	    c='r')

ax2 = fig.add_subplot(132)
wages.plot(x='year', y='wage', kind='scatter', ax=ax2, alpha=0.5)
ax2.set_xlabel('Year')
ax2.set_ylabel('')
year_wage_reg = ols(formula='wage ~ year', data=wages)
year_wage_fit = year_wage_reg.fit()
ax2.scatter(x=np.array(wages['year']), y=year_wage_fit.predict(), marker='+',
	    c='r')

ax3 = fig.add_subplot(133)
wages.boxplot(column='wage', by='ed_level', ax=ax3, grid=False)
ax3.set_title('')
ax3.set_xlabel('Education level')

fig.suptitle('')
fig.tight_layout()
