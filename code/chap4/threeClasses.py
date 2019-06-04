import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import multivariate_normal

np.random.seed(911)
X = np.random.normal(size=(150, 2))
y = np.zeros(150, dtype=int)
X[:50, :] = X[:50, :] - 2
X[50:100, :] = X[50:100, :] + 2
y[50:100] = y[50:100] + 1
X[100:, 0] = X[100:, 0] - 2
X[100:, 1] = X[100:, 1] + 2
y[100:] = y[100:] + 2

gnb = GaussianNB()
gnb.fit(X, y)

clf = LDA()
clf.fit(X, y)

x1_array = np.linspace(-5, 5, 200)
x2_array = np.linspace(-5, 5, 200)
x1_grid, x2_grid = np.meshgrid(x1_array, x2_array)
z_gnb_array = gnb.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
z_gnb_grid = np.reshape(z_gnb_array, x1_grid.shape)
prob_gnb = gnb.predict_proba(
    np.column_stack((x1_grid.ravel(), x2_grid.ravel())))


z_lda_array = clf.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
z_lda_grid = np.reshape(z_lda_array, x1_grid.shape)
# prob_lda = clf.predict_proba(
#     np.column_stack((x1_grid.ravel(), x2_grid.ravel())))

# rv0 = multivariate_normal(mean=[-2, -2], cov=[[1, 0], [0, 1]])
# rv1 = multivariate_normal(mean=[2, 2], cov=[[1, 0], [0, 1]])
# rv2 = multivariate_normal(mean=[-2, 2], cov=[[1, 0], [0, 1]])

# prob0_array = rv0.cdf(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
# prob0_grid = np.reshape(prob0_array, x1_grid.shape)

pcol = ['green', 'blue', 'brown']
pch = ['+', 'x', 'o']

X_new = np.random.normal(size=(60, 2))
y_new = np.zeros(60, dtype=int)
X_new[:20, :] = X_new[:20, :] - 2
X_new[20:40, :] = X_new[20:40, :] + 2
y_new[20:40] = y_new[20:40] + 1
X_new[40:, 0] = X_new[40:, 0] - 2
X_new[40:, 1] = X_new[40:, 1] + 2
y_new[40:] = y_new[40:] + 2

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.contour(x1_grid, x2_grid, z_gnb_grid, linestyles='dashed')
ax1.set_xlabel(r'$X_1$')
ax1.set_ylabel(r'$X_2$')

ax2 = fig.add_subplot(122)
ax2.contour(x1_grid, x2_grid, z_gnb_grid, linestyles='dashed')
ax2.contour(x1_grid, x2_grid, z_lda_grid)
for i in range(3):
    ax2.scatter(X_new[y_new == i, 0], X_new[y_new == i, 1],
                marker=pch[i], color=pcol[i], alpha=0.7)
ax2.set_xlabel(r'$X_1$')
ax2.set_ylabel(r'$X_2$')

fig.tight_layout()
