import numpy as np
from sklearn import neighbors, naive_bayes
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


# A simple function to assign two categories
# def nameX(x1, x2):
#     if x2 < 0.33:
#         name = 0
#     elif x2 < 0.66:
#         if x1 < 0.33:
#             name = 1
#         else:
#             name = 0
#     else:
#         if x1 < 0.66:
#             name = 1
#         else:
#             name = 0
#     return name
def nameX(x1, x2):
    if x2 < 0.2:
        name = 0
    elif x2 < 0.4:
        if (0.2 < x1 and  x1 < 0.4) or x1 > 0.6:
            name = 0
        else:
            name = 1
    elif x2 < 0.6:
        if x1 < 0.6:
            name = 1
        else:
            name = 0
    elif x2 < 0.8:
        if x1 < 0.4:
            name = 1
        else:
            name = 0
    else:
        if x1 < 0.8:
            name = 1
        else:
            name = 0
    return name


def plotClassify(Bayes=True, KNN=False, region='Bayes', k_neighbors=10,
                 plot_title=False):
    # Generate data used to fit KNN
    np.random.seed(911)
    x1 = np.linspace(0, 1, 20)
    x2 = np.linspace(0, 1, 10)
    xx1, xx2 = np.meshgrid(x1, x2)

    xx1 = xx1 + np.random.normal(scale=0.1, size=xx1.shape)
    xx2 = xx2 + np.random.normal(scale=0.1, size=xx2.shape)
    name_x = np.vectorize(nameX)(xx1.ravel(), xx2.ravel())
    name_x = name_x.reshape(xx1.shape)

    clf = neighbors.KNeighborsClassifier(k_neighbors, weights='uniform')
    clf.fit(np.c_[xx1.ravel(), xx2.ravel()], name_x.ravel())
    gnb = naive_bayes.GaussianNB()
    bayes_fit = gnb.fit(np.c_[xx1.ravel(), xx2.ravel()], name_x.ravel())

    name_predict = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    name_predict = name_predict.reshape(xx1.shape)

    # More granular data to show region, plot boundary
    xx = np.linspace(xx1.min() - 0.1, xx1.max() + 0.1)
    yy = np.linspace(xx2.min() - 0.1, xx2.max() + 0.1)
    xx_mesh, yy_mesh = np.meshgrid(xx, yy)
    zz = clf.predict(np.c_[xx_mesh.ravel(), yy_mesh.ravel()])
    zz_mesh = zz.reshape(xx_mesh.shape)
    zz_bayes = bayes_fit.predict(np.c_[xx_mesh.ravel(), yy_mesh.ravel()])
    zz_bayes_mesh = zz_bayes.reshape(xx_mesh.shape)

    if region == 'KNN':
        plt.scatter(xx_mesh, yy_mesh, s=1, c=zz_mesh, alpha=0.5)
    if region == 'Bayes':
        plt.scatter(xx_mesh, yy_mesh, s=1, c=zz_bayes_mesh, alpha=0.5)
    if Bayes is True:
        plt.contour(xx_mesh, yy_mesh, zz_bayes_mesh, linestyles='dashed',
                    levels=[0, 1])
    if KNN is True:
        plt.contour(xx_mesh, yy_mesh, zz_mesh, levels=[0, 1])
    plt.scatter(xx1, xx2, c=name_x, marker='x', s=15)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    if plot_title is True:
        plt.title('KNN: K={}'.format(k_neighbors))
