# ISLR functions used to plot figure 5.6

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit, polyval


def generateDataSets(true_func, x_min=0, x_max=100, df_size=50, error_sd=1.0):
    my_df = pd.DataFrame({
        'x': np.linspace(x_min, x_max, num=2 * df_size)})
    my_df['y_true'] = my_df.apply(true_func)
    my_df['y_observe'] = my_df['y_true'] + \
        np.random.normal(loc=0, scale=error_sd, size=my_df.shape[0])
    train_test_ind = np.random.choice(2, size=my_df.shape[0])
    train_df = my_df.loc[train_test_ind == 0]
    test_df = my_df.loc[train_test_ind == 1]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return (train_df, test_df)


def trueTestMSE(train_df, test_df, max_deg=15):
    degree = []
    mse_test = []
    # pdb.set_trace()
    for i_degree in range(1, max_deg+1):
        poly_fit = polyfit(train_df['x'], train_df['y_observe'],
                           deg=i_degree)
        y_predict = polyval(test_df['x'], poly_fit)
        mse = np.mean((y_predict - test_df['y_observe']) ** 2)
        degree.append(i_degree)
        mse_test.append(mse)
    return (degree, mse_test)


def loocvMSE(train_df, max_deg=15):
    mse_degree = []
    degree = []
    for i_degree in range(1, max_deg + 1):
        mse_deg = []
        for i in range(train_df.shape[0]):
            test_ind = (np.array(train_df.index) != i)
            train_data = train_df.loc[test_ind]
            poly_fit = polyfit(
                train_data['x'], train_data['y_observe'], deg=i_degree)
            y_predict = polyval(train_df.iloc[i]['x'], poly_fit)
            mse_deg.append(
                (y_predict - train_df.iloc[i]['y_observe']) ** 2)
        mse_degree.append(np.mean(mse_deg))
        degree.append(i_degree)
    return (degree, mse_degree)


def kFoldCV_MSE(train_df, n_folds=10, max_deg=15):
    mse_folds = {}
    for i_degree in range(1, max_deg + 1):
        mse_folds[i_degree] = []
    fold_ind = np.random.choice(n_folds, train_df.shape[0])
    for i_degree in range(1, max_deg + 1):
        for i_fold in range(n_folds):
            train_ind = (fold_ind != i_fold)
            test_ind = (fold_ind == i_fold)
            test_data = train_df.loc[test_ind]
            train_data = train_df.loc[train_ind]
            poly_fit = polyfit(train_data['x'], train_data['y_observe'],
                               deg=i_degree)
            y_predict = polyval(test_data['x'], poly_fit)
            mse = np.mean((y_predict - test_data['y_observe']) ** 2)
            mse_folds[i_degree].append(mse)
    mse_degree = []
    degree = []
    for i_degree in mse_folds.keys():
        mse_degree.append(np.mean(mse_folds[i_degree]))
        degree.append(i_degree)
    return (degree, mse_degree)
