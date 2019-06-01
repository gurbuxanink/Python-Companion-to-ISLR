# Figure 4.3

import matplotlib.pyplot as plt
from statsmodels import datasets
import pandas as pd
import numpy as np

default = datasets.get_rdataset('Default', 'ISLR').data
default['balance_grp'] = pd.cut(default['balance'], bins=np.linspace(0, 2700, 10))

student_balance = default.loc[default['student'] == 'Yes', ['balance', 'balance_grp']].groupby('balance_grp').mean()
student_defrate = default.loc[default['student'] == 'Yes'].groupby('balance_grp').apply(lambda x: np.sum(x['default'] == 'Yes') / x.shape[0])
student_defrate.name = 'default_rate'
student_data = pd.merge(student_balance, student_defrate, left_index=True,
                        right_index=True)

notstudent_balance = default.loc[default['student'] == 'No', ['balance', 'balance_grp']].groupby('balance_grp').mean()
notstudent_defrate = default.loc[default['student'] == 'No'].groupby('balance_grp').apply(lambda x: np.sum(x['default'] == 'Yes') / x.shape[0])
notstudent_defrate.name = 'default_rate'
notstudent_data = pd.merge(notstudent_balance, notstudent_defrate,
                           left_index=True, right_index=True)

overall_default = default.groupby('student').apply(lambda x: np.sum(x['default'] == 'Yes') / x.shape[0])

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
student_data.plot(x='balance', y='default_rate', color='brown',
                  label='Student', ax=ax1)
notstudent_data.plot(x='balance', y='default_rate', c='b', linestyle='--',
                     label='Not Student', ax=ax1)
ax1.axhline(y=overall_default['No'], c='b', linestyle='--', linewidth=0.7)
ax1.axhline(y=overall_default['Yes'], color='brown', linewidth=0.7)
ax1.set_xlabel('Credit Card Balance')
ax1.set_ylabel('Default Rate')

ax2 = fig.add_subplot(122)
default.boxplot(column='balance', by='student', grid=False, widths=0.6, ax=ax2)
ax2.set_xlabel('Student Status')
ax2.set_ylabel('Credit Card Balance')
ax2.set_title('')

fig.suptitle('')
fig.tight_layout()
