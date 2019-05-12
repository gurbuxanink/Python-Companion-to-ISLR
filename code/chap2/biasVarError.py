# Inputs for ISLR figure 2.12
# Given a true function, simulate observed data
# Estimate model parameters for different degrees of freedom
# Return MSE, bias, and variance

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
# import pdb


def mseBiasVar(true_func, x_min=0, x_max=100, df_size=50, error_sd=1.0,
               num_trials=40):
    degree = []
    mse = []
    var = []
    bias = []
    for i_degree in range(1, 16):
        # pdb.set_trace()
        my_df = pd.DataFrame({
            'x': np.linspace(x_min, x_max, num=2*df_size)})
        my_df['y_true'] = my_df['x'].apply(true_func)
        train_ind = random.sample(range(2*df_size), df_size)
        train_ind.sort()
        test_ind = set(range(2*df_size)) - set(train_ind)
        test_ind = list(test_ind)
        test_ind.sort()
        y_fit_trials = np.zeros((df_size, num_trials))
        mse_deg = []

        for j_trial in range(num_trials):
            my_df['y_observe'] = my_df['y_true'] + \
                np.random.normal(loc=0, scale=error_sd, size=2*df_size)
            train_df = my_df.iloc[train_ind]
            test_df = my_df.iloc[test_ind]

            poly_fit = np.polyfit(train_df['x'], train_df['y_observe'],
                                  deg=i_degree)
            y_predict = np.polyval(poly_fit, test_df['x'])
            y_fit_trials[:, j_trial] = y_predict
            mse_deg.append(np.sum((y_predict - train_df['y_observe']) ** 2) /
                           df_size)

        y_avg_fit = y_fit_trials.mean(axis=1)
        # bias squared
        my_bias = np.sum((y_avg_fit - test_df['y_true'])**2) / df_size
        y_fit_adj = y_fit_trials - np.reshape(y_avg_fit, (df_size, 1))
        y_fit_adj_sq = y_fit_adj ** 2
        y_trial_var = y_fit_adj_sq.mean(axis=1)
        my_var = np.sum(y_trial_var) / df_size
        degree.append(i_degree)
        bias.append(my_bias)
        var.append(my_var)
        # mse.append(my_var + my_bias)
        mse.append(np.sum(mse_deg) / num_trials)

    return (degree, mse, var, bias)


def plotBiasVarMse(degree, mse, var, bias, include_legend=False):
    plt.plot(degree, mse, color='brown')
    plt.plot(degree, var, color='orange', linestyle='--')
    plt.plot(degree, bias, color='blue', linestyle='-.')
    plt.axhline(y=1, color='grey', linestyle='--')
    min_mse_ind = mse.index(min(mse))
    plt.axvline(x=degree[min_mse_ind], color='grey', linestyle=':')
