# Three dimensional setting with two predictors and one response
# We simulate data used in plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf

np.random.seed(911)
x1 = np.random.uniform(0, 5, 50)
x2 = np.random.uniform(0, 5, x1.size)
y = x1 + x2 + np.random.normal(loc=0, scale=4, size=x1.size)

my_df = pd.DataFrame({'x1': x1 , 'x2': x2, 'y': y})
reg_model = smf.ols(formula='y ~ x1 + x2', data=my_df)
reg_fit = reg_model.fit()
y_fit = reg_fit.fittedvalues

x1_array = np.linspace(0, 5, 20)
x2_array = np.linspace(0, 5, 20)
x1_grid, x2_grid = np.meshgrid(x1_array, x2_array)
y_series = reg_fit.predict(exog=dict(x1=x1_grid.ravel(),
                                    x2=x2_grid.ravel()))
y_grid = y_series.values.reshape(x1_grid.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x1_grid, x2_grid, y_grid, alpha=0.3)
for i in range(x1.size):
    ax.plot([x1[i], x1[i]], [x2[i], x2[i]], [y_fit[i], y[i]], color='grey')
ax.scatter(x1, x2, y, c='r')
ax.set_xlabel(r'$X_1$')
ax.set_ylabel(r'$X_2$')
ax.set_zlabel(r'$Y$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
fig.tight_layout()
