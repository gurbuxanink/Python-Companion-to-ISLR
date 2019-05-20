import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import statsmodels.formula.api as smf
import numpy as np

advertising = pd.read_csv('data/Advertising.csv', index_col=0)
sales_tv_model = smf.ols(formula='sales ~ TV', data=advertising)
sales_tv_fit = sales_tv_model.fit()

beta0 = sales_tv_fit.params['Intercept']
beta1 = sales_tv_fit.params['TV']

beta0_xx = np.linspace(0.5 * beta0, 1.5 * beta0)
beta1_xx = np.linspace(0.5 * beta1, 1.5 * beta1)
beta0_grid, beta1_grid = np.meshgrid(beta0_xx, beta1_xx)


def calcRSS(b0, b1):
    sales_pred = b0 + b1 * advertising['TV']
    error = sales_pred - advertising['sales']
    return np.sum(error ** 2)


rss = np.vectorize(calcRSS)(beta0_grid.ravel(), beta1_grid.ravel())
rss_grid = np.reshape(rss, beta0_grid.shape)

ctr_levels = [2100, 2200, 2300, 2500, 3000, 3500]
ctr_labels = [str(level / float(1000)) for level in ctr_levels]
ctr_fmt = {}
for level, fmt in zip(ctr_levels, ctr_labels):
    ctr_fmt[level] = fmt

fig = plt.figure()
ax1 = fig.add_subplot(121)
CS = ax1.contour(beta0_grid, beta1_grid, rss_grid, colors='blue',
                 levels=ctr_levels, linewidth=0.5)
ax1.clabel(CS, CS.levels, inline=True, fmt=ctr_fmt)
ax1.scatter(beta0, beta1, marker='o', s=100, c='r')
ax1.set_xlabel(r'$\beta_0$')
ax1.set_ylabel(r'$\beta_1$')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_wireframe(beta0_grid, beta1_grid, rss_grid, rstride = 5,
                   cstride = 5, alpha=0.7)
ax2.scatter(beta0, beta1, sales_tv_fit.ssr, marker='o', c='r', s=100)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.set_xlabel(r'$\beta_0$')
ax2.set_ylabel(r'$\beta_1$')
ax2.set_zlabel('RSS', rotation=90)
ax2.grid(False)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

fig.tight_layout()
