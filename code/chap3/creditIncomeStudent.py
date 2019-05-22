import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.datasets as datasets

credit = datasets.get_rdataset('Credit', 'ISLR').data

inc_stud_model = smf.ols(formula='Balance ~ Income + Student', data=credit)
inc_stud_fit = inc_stud_model.fit()
credit['balance_fit_all'] = inc_stud_fit.fittedvalues
income_x = np.linspace(0, credit['Income'].max())
balance_fit_stud = inc_stud_fit.predict(
    exog=dict(Income=income_x, Student=np.repeat(['Yes'], income_x.size)))
balance_fit_not_stu = inc_stud_fit.predict(
    exog=dict(Income=income_x, Student=np.repeat(['No'], income_x.size)))

credit_student = credit.loc[credit['Student'] == 'Yes'].copy()
credit_not_student = credit.loc[credit['Student'] == 'No'].copy()

stud_model = smf.ols(formula='Balance ~ Income', data=credit_student)
stud_fit = stud_model.fit()
credit_student['balance_fit_student'] = stud_fit.fittedvalues
balance_fit_stud_separate = stud_fit.predict(exog=dict(Income=income_x))

not_stud_model = smf.ols(formula='Balance ~ Income', data=credit_not_student)
not_stud_fit = not_stud_model.fit()
credit_not_student['balance_fit_not_stud'] = not_stud_fit.fittedvalues
balance_fit_not_stud_separate = not_stud_fit.predict(exog=dict(Income=income_x))

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.plot(income_x, balance_fit_stud.values, c='r')
ax1.plot(income_x, balance_fit_not_stu, c='k', linestyle='--')
ax1.set_xlabel('Income')
ax1.set_ylabel('Balance')
ax1.set_ylim(credit['Balance'].min(), credit['Balance'].max())

ax2 = fig.add_subplot(122)
ax2.plot(income_x, balance_fit_stud_separate, c='r', label='student')
ax2.plot(income_x, balance_fit_not_stud_separate, c='k', linestyle='--',
         label='non-student')
ax2.set_ylim(credit['Balance'].min(), credit['Balance'].max())
ax2.set_xlabel('Income')
ax2.legend()

fig.tight_layout()
