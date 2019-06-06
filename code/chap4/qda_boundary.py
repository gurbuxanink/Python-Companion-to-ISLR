# Figure 4.9

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB

avg_brown = [-1.5, -1.5]
avg_blue = [1, 1]
cor_brown = [0.7, 0.7]
cor_blue = [0.7, -0.7]
sample_size = 50

y = np.append(np.zeros(sample_size, dtype='int'),
              np.ones(sample_size, dtype=int))

np.random.seed(911)
fig = plt.figure(figsize=(8, 4))

for i in range(2):
    X_brown = np.random.multivariate_normal(mean=avg_brown,
                                            cov=[[1, cor_brown[i]],
                                                 [cor_brown[i], 1]],
                                            size=sample_size)
    X_blue = np.random.multivariate_normal(mean=avg_blue,
                                           cov=[[1, cor_blue[i]],
                                                [cor_blue[i], 1]],
                                           size=sample_size)
    X = np.row_stack((X_brown, X_blue))

    gnb = GaussianNB()
    gnb.fit(X, y)

    clf = LDA()
    clf.fit(X, y)

    qdf = QDA()
    qdf.fit(X, y)

    x1_min, x2_min = X.min(axis=0)
    x1_max, x2_max = X.max(axis=0)
    x1_array = np.linspace(x1_min, x1_max, 400)
    x2_array = np.linspace(x2_min, x2_max, 400)
    x1_grid, x2_grid = np.meshgrid(x1_array, x2_array)

    z_gnb = gnb.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
    z_gnb_grid = np.reshape(z_gnb, x1_grid.shape)

    z_clf = clf.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
    z_clf_grid = np.reshape(z_clf, x1_grid.shape)

    z_qdf = qdf.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
    z_qdf_grid = np.reshape(z_qdf, x1_grid.shape)

    ax = fig.add_subplot(1, 2, i+1)
    ax.contour(x1_grid, x2_grid, z_gnb_grid,
               linestyles='dashed', colors='purple')
    ax.contour(x1_grid, x2_grid, z_clf_grid,
               linestyles='dotted', colors='black')
    ax.contour(x1_grid, x2_grid, z_qdf_grid, colors='green')
    ax.scatter(X_brown[:, 0], X_brown[:, 1],
               marker='+', color='brown', alpha=0.7)
    ax.scatter(X_blue[:, 0], X_blue[:, 1], marker='x', color='blue', alpha=0.7)
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')

fig.tight_layout()
