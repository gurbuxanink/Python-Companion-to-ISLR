# alphaBootstrap.py

import numpy as np
import pandas as pd
# from statsmodels import datasets
import statsmodels.formula.api as smf


def alphaEst(returns_df, row_index):
    '''Assumes returns_df is a return dataframe with two columns of stock returns,
    row_index is a list of row indexes to be used in calculation.
    Returns alpha estimate using subset of data defined by row_index.'''

    cov_xy = np.cov(returns_df.iloc[row_index], rowvar=False)
    return (cov_xy[1, 1] - cov_xy[0, 1]) / \
        (cov_xy[0, 0] + cov_xy[1, 1] - 2 * cov_xy[0, 1])


def bootStrap(my_df, myFunc, sample_size, n_bootstrap, all_res=False):
    ''' Assumes my_df is a dataframe and myFunc is a function that can
    estimate a stastic on my_df.  Estimate statistic n_bootstrap times,
    each with a sample of size sample_size.
    Return mean and standard error of statistic.'''
    my_stat = []
    for i in range(n_bootstrap):
        index = np.random.choice(my_df.shape[0], sample_size)
        my_stat.append(myFunc(my_df, index))

    if isinstance(my_stat[0], float):
        my_res = {'mean': np.mean(my_stat), 'std. error': np.std(my_stat)}
        if all_res:
            my_res['stats'] = my_stat

    elif isinstance(my_stat[0], pd.core.series.Series):
        my_stat_dict = {}
        for ind in my_stat[0].index:
            my_stat_dict[ind] = []
        for i in range(len(my_stat)):
            for key in my_stat_dict.keys():
                my_stat_dict[key].append(my_stat[i][key])
        my_res = {}
        for key in my_stat_dict.keys():
            my_res[key] = {}
            my_res[key]['mean'] = np.mean(my_stat_dict[key])
            my_res[key]['std. error'] = np.std(my_stat_dict[key])
        if all_res:
            my_res['stats'] = my_stat

    return my_res


def autoDataCoef(auto_df, row_index):
    '''Assumes auto_df is a dataframe which includes 'mpg' and
    'horsepower' columns.  Fit a linear regression model on auto_df.
    Use row_index to create a subset of auto_df.  Return regression
    coefficients estimated from subset of auto_df.'''
    lm_model = smf.ols('mpg ~ horsepower', data=auto_df.iloc[row_index])
    lm_fit = lm_model.fit()
    return lm_fit.params


def autoDataCoef2(auto_df, row_index):
    '''Assumes auto_df is a dataframe which has columns 'mpg' and
    'horsepower'.  Fit an OLS regression model with mpg as a 
    quadratic function of horsepower.  Use subset of auto_df defined
    by row_index.  Return regression coefficient estimates.'''
    lm_model = smf.ols('mpg ~ horsepower + I(horsepower ** 2)',
                       data=auto_df.iloc[row_index])
    lm_fit = lm_model.fit()
    return lm_fit.params
