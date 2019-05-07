# Read Income2.csv data
# Plot three dimensional graph of fit surface and data points
# Function has one input: fit formula

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols
from mpl_toolkits.mplot3d import Axes3D


def plotIncomeEdSeniority(fit_formula):
    '''Assumes: fit_formula is a valid formula for ols in statsmodels
    Returns: 3D graph of Income versus Education and Seniority'''

    # pdb.set_trace()
    income_data = pd.read_csv('data/Income2.csv', index_col=0)

    reg = ols(formula=fit_formula, data=income_data)
    reg_fit = reg.fit()
    income_data['income_predict'] = reg_fit.predict()

    ed_rng = np.linspace(income_data['Education'].min(),
                         income_data['Education'].max())
    sen_rng = np.linspace(income_data['Seniority'].min(),
                          income_data['Seniority'].max())
    ed, sen = np.meshgrid(ed_rng, sen_rng)
    ed_array = np.reshape(ed, newshape=-1)
    sen_array = np.reshape(sen, newshape=-1)
    income_fit = reg_fit.predict(exog=dict(Education=ed_array,
                                           Seniority=sen_array))
    income_fit = np.reshape(income_fit, newshape=ed.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(ed, sen, income_fit, alpha=0.5, rstride=5, cstride=5)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_zticklabels('')
    ax.set_xlabel(r'$\overrightarrow{Years of Education}$')
    ax.set_ylabel(r'$\overrightarrow{Seniority}$')
    ax.set_zlabel(r'$\overrightarrow{Income}$')

    ax.scatter(xs=income_data['Education'], ys=income_data['Seniority'],
               zs=income_data['Income'], c='r')

    for i in range(income_data.shape[0]):
        x = income_data.iloc[i]['Education']
        y = income_data.iloc[i]['Seniority']
        z1 = income_data.iloc[i]['income_predict']
        z2 = income_data.iloc[i]['Income']
        ax.plot([x, x], [y, y], [z1, z2], c='b', alpha=0.5)

        fig.tight_layout()
