# ISLR figures 2.9, 2.10, and 2.11
# For left panel, use a function that looks similar to the book figure
# Since SciPy UnivariateSpline does not have more than 5 degrees,
# for right panel, use polynomial fits, not splines

from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


def mseVsFlexibility(true_func, x_min=0, x_max=100, df_size=50, error_sd=1.0):
    my_df = pd.DataFrame({
        'x': np.linspace(x_min, x_max, num=2*df_size)})
    my_df['y_true'] = my_df['x'].apply(true_func)
    my_df['y_observe'] = my_df['y_true'] + \
        np.random.normal(loc=0, scale=error_sd, size=2*df_size)
    train_ind = random.sample(range(2*df_size), df_size)
    train_ind.sort()
    test_ind = set(range(2 * df_size)) - set(train_ind)
    test_ind = list(test_ind)
    train_df = my_df.iloc[train_ind]
    test_df = my_df.iloc[test_ind]

    poly_1d_fit = np.polyfit(train_df['x'], train_df['y_observe'], deg=1)
    train_df['y_linear_fit'] = train_df['x'].apply(
        lambda u: np.polyval(poly_1d_fit, u))

    spline_5d = UnivariateSpline(train_df['x'], train_df['y_observe'], k=5)
    train_df['y_spline_5d'] = train_df['x'].apply(spline_5d)

    spline_5d_knots = UnivariateSpline(train_df['x'], train_df['y_observe'],
                                       k=5, s=6)
    train_df['y_spline_5d_knots'] = train_df['x'].apply(spline_5d_knots)

    fig = plt.figure()
    # Left panel: true function, observed values scatter plot, three fits
    ax1 = fig.add_subplot(121)
    train_df.plot(x='x', y='y_true', legend=False, ax=ax1, c='k', alpha=0.7)
    train_df.plot(x='x', y='y_observe', kind='scatter', legend=False, ax=ax1,
                  marker='o', alpha=0.7, facecolors='none')
    train_df.plot(x='x', y='y_linear_fit', legend=False, ax=ax1,
                  linestyle='--', color='orange')
    train_df.plot(x='x', y='y_spline_5d', legend=False, ax=ax1, linestyle='-.',
                  c='g')
    train_df.plot(x='x', y='y_spline_5d_knots', legend=False, ax=ax1,
                  linestyle=':', c='b', alpha=0.7)
    ax1.set_ylabel('Y')
    ax1.set_xlabel('X')

    # Right panel: train and test mse versus flexibility
    # Use polynomial degree as a proxy for flexibility
    degree = []
    mse_train = []
    mse_test = []
    for i_degree in range(1, 16):
        poly_fit = np.polyfit(train_df['x'], train_df['y_observe'],
                              deg=i_degree)
        y_predict_train = np.polyval(poly_fit, train_df['x'])
        y_predict_test = np.polyval(poly_fit, test_df['x'])
        train_var = np.var(train_df['y_observe'] - y_predict_train)
        test_var = np.var(test_df['y_observe'] - y_predict_test)
        degree.append(i_degree)
        mse_train.append(train_var)
        mse_test.append(test_var)

        ax2 = fig.add_subplot(122)
        ax2.plot(degree, mse_train, color='grey')
        ax2.plot(degree, mse_test, color='red', linestyle='-.')
        ax2.set_xlabel('Flexibility')
        ax2.set_ylabel('Mean Squared Error')
        ax2.axhline(y=error_sd, linestyle='--', color='grey')

        fig.tight_layout()
