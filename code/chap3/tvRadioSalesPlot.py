import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf
import numpy as np

advertising = pd.read_csv('data/Advertising.csv', index_col=0)
reg_model = smf.ols(formula='sales ~ TV + radio', data=advertising)
reg_fit = reg_model.fit()
tv = advertising['TV'].values
radio = advertising['radio'].values
sales = advertising['sales'].values
sales_fit = reg_fit.fittedvalues.values

tv_array = np.linspace(0.9 * advertising['TV'].min(),
		       1.1 * advertising['TV'].max())
radio_array = np.linspace(0.9 * advertising['radio'].min(),
			  1.1 * advertising['radio'].max())
tv_grid, radio_grid = np.meshgrid(tv_array, radio_array)
sales_pred = reg_fit.predict(exog=dict(TV=tv_grid.ravel(),
				       radio=radio_grid.ravel()))
sales_grid = sales_pred.values.reshape(tv_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(tv_grid, radio_grid, sales_grid, alpha=0.3,
		  rcount=10, ccount=10)
for i in range(len(tv)):
    ax.plot([tv[i], tv[i]], [radio[i], radio[i]],
	    [sales_fit[i], sales[i]], color='grey')
ax.scatter(advertising['TV'], advertising['radio'],
	   advertising['sales'], c='r')
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Sales')
fig.tight_layout()
