# Figure 5.7

from knnBoundry import nameX
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.naive_bayes import GaussianNB
sys.path.append('code/chap2/')


# Generate data
np.random.seed(411)
train_size = 100
error_sd = 0.05
my_df = pd.DataFrame({'x1': np.random.uniform(size=train_size),
                      'x2': np.random.uniform(size=train_size)})
my_df['y'] = my_df.apply(lambda row: nameX(row['x1'], row['x2']), axis=1)
my_df['x1'] = my_df['x1'] + np.random.normal(scale=error_sd, size=train_size)
my_df['x2'] = my_df['x2'] + np.random.normal(scale=error_sd, size=train_size)
my_df_grp0 = my_df.loc[my_df['y'] == 0]
my_df_grp1 = my_df.loc[my_df['y'] == 1]


gnb = GaussianNB()
bayes_fit = gnb.fit(my_df[['x1', 'x2']], my_df['y'])

# More granular data to show region and boundary
xx = np.linspace(my_df['x1'].min() - 0.1, my_df['x1'].max() + 0.1, 500)
yy = np.linspace(my_df['x2'].min() - 0.1, my_df['x2'].max() + 0.1, 500)
xx_mesh, yy_mesh = np.meshgrid(xx, yy)
zz_bayes = bayes_fit.predict(
    np.column_stack((xx_mesh.ravel(), yy_mesh.ravel())))
zz_bayes_mesh = zz_bayes.reshape(xx_mesh.shape)

my_formula = 'y ~ x1 + x2'
fig = plt.figure()
for i_degree in range(1, 5):
    ax = fig.add_subplot(2, 2, i_degree)
    ax.contour(xx_mesh, yy_mesh, zz_bayes_mesh, linestyles='dashed',
               colors='blue')
    my_df_grp0.plot(x='x1', y='x2', kind='scatter', marker='x', c='b',
                    alpha=0.7, ax=ax)
    my_df_grp1.plot(x='x1', y='x2', kind='scatter', marker='+', c='r',
                    alpha=0.7, ax=ax)
    logit_fit = smf.logit(formula=my_formula, data=my_df).fit()
    zz_logit_prob = logit_fit.predict(exog=dict(x1=xx_mesh.ravel(),
                                                x2=yy_mesh.ravel()))
    zz_logit = np.vectorize(int)(zz_logit_prob > 0.5)
    zz_logit_mesh = zz_logit.reshape(xx_mesh.shape)
    ax.contour(xx_mesh, yy_mesh, zz_logit_mesh, colors='black')
    my_formula += ' + I(x1 ** ' + str(i_degree + 1) + ') + I(x2 ** ' + \
        str(i_degree + 1) + ')'

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Degree = ' + str(i_degree))

fig.tight_layout()
