# Plot figure 3.17

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsRegressor

def trueFunc(x):
    return 2 * x + 2
error_sd = 0.2

np.random.seed(911)
my_df = pd.DataFrame({'x': np.random.uniform(-1, 1, size=50)})
my_df['y'] = np.vectorize(trueFunc)(my_df['x']) + \
    np.random.normal(loc=0.0, scale=error_sd, size=my_df.shape[0])

knn_reg = KNeighborsRegressor(n_neighbors=1)
knn_reg.fit(my_df['x'].values.reshape(-1, 1), my_df['y'])

x_array = np.linspace(-1, 1, 400)
y_knn1 = knn_reg.predict(x_array.reshape(-1, 1))
y_true = np.vectorize(trueFunc)(x_array)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
my_df.plot(x='x', y='y', kind='scatter', c='r', alpha=0.7, s=7,  ax=ax1)
ax1.plot(x_array, y_knn1, c='b', alpha=0.7)
ax1.plot(x_array, y_true, c='k', alpha=0.7, linestyle='--')

knn_reg9 = KNeighborsRegressor(n_neighbors=9)
knn_reg9.fit(my_df['x'].values.reshape(-1, 1), my_df['y'])
y_knn9 = knn_reg9.predict(x_array.reshape(-1, 1))

ax2 = fig.add_subplot(122)
my_df.plot(x='x', y='y', kind='scatter', c='r', alpha=0.7, s=7, ax=ax2)
ax2.plot(x_array, y_knn9, c='b', alpha=0.7)
ax2.plot(x_array, y_true, c='k', alpha=0.7, linestyle='--')

fig.tight_layout()
