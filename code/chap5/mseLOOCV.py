# mseLOOCV.py

import numpy as np
from statsmodels import datasets
import statsmodels.formula.api as smf

auto = datasets.get_rdataset('Auto', 'ISLR').data
all_ind = np.arange(auto.shape[0])

my_formula = 'mpg ~ horsepower'

mse_loocv = []
degree = []
for i_degree in range(1, 6):
    mse = []
    for i_obs in range(auto.shape[0]):
        # auto_train = auto.loc[all_ind != i_obs]
        auto_train = auto.drop(auto.index[i_obs])
        auto_test = auto.iloc[i_obs]
        lm_model = smf.ols(my_formula, data=auto_train)
        lm_fit = lm_model.fit()
        hp_predict = lm_fit.predict(
            exog=dict(horsepower=auto_test['horsepower']))
        mse.append((hp_predict - auto_test['mpg']) ** 2)

    mse_loocv.append(np.mean(mse))
    degree.append(i_degree)
    my_formula += ' + I(horsepower **' + str(i_degree + 1) + ')'

for i_degree, mse in zip(degree, mse_loocv):
    print('degree: ', i_degree, ', mse_loocv:', round(mse, 3))
