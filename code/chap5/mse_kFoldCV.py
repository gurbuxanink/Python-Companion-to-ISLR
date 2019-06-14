# mse_kFoldCV.py

import numpy as np
import statsmodels.formula.api as smf
from statsmodels import datasets

auto = datasets.get_rdataset('Auto', 'ISLR').data

n_folds = 10
max_degree = 10

np.random.seed(911)
fold_ind = np.random.choice(n_folds, auto.shape[0])
all_ind = np.arange(auto.shape[0])
degree = []
mse_folds = {}

my_formula = 'mpg ~ horsepower'
for i_degree in range(1, max_degree + 1):
    mse_folds[i_degree] = []
    for i_fold in range(n_folds):
        train_df = auto.loc[i_fold != fold_ind]
        test_df = auto.loc[i_fold == fold_ind]
        lm_model = smf.ols(my_formula, data=train_df)
        lm_fit = lm_model.fit()
        mse = np.mean((lm_fit.predict(test_df) - test_df['mpg']) ** 2)
        mse_folds[i_degree].append(mse)

    degree.append(i_degree)
    my_formula += ' + I(horsepower ** ' + str(i_degree + 1) + ')'

mse_degree = []
for i_degree in mse_folds.keys():
    mse_degree.append(np.mean(mse_folds[i_degree]))

for i_degree, mse_kfold in zip(degree, mse_degree):
    print('degree: ', i_degree, ', mse_kfold: ', round(mse_kfold, 3))
