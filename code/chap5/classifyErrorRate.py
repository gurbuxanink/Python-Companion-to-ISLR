# Figure 5.8
# Plot error rates for 10-fold CV and KNN classifier

from knnBoundry import nameX
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.formula.api as smf
import sys
import pandas as pd
sys.path.append('code/chap2/')

np.random.seed(911)


def generateDF(trueFunc, df_size=100, x_min=0.0, x_max=1.0, error_sd=0.05):
    my_df = pd.DataFrame({'x1': np.random.uniform(low=x_min, high=x_max,
                                                  size=df_size),
                          'x2': np.random.uniform(low=x_min, high=x_max,
                                                  size=df_size)})
    my_df['y'] = my_df.apply(
        lambda row: trueFunc(row['x1'], row['x2']), axis=1)
    my_df['x1'] = my_df['x1'] + np.random.normal(scale=error_sd, size=df_size)
    my_df['x2'] = my_df['x2'] + np.random.normal(scale=error_sd, size=df_size)

    return my_df


train_df = generateDF(nameX, df_size=200)
test_df = generateDF(nameX, df_size=200)

# Error rates with logit regression on polynomials of x1 and x2
my_formula = 'y ~ x1 + x2'
max_poly_degree = 10
poly_degree = []
error_test = []
error_train = []
n_folds = 10
error_fold = {}
for i_degree in range(1, max_poly_degree + 1):
    error_fold[i_degree] = []
fold_ind = np.random.choice(n_folds, train_df.shape[0])

for i_degree in range(1, max_poly_degree + 1):
    logit_model = smf.logit(my_formula, data=train_df)
    logit_fit = logit_model.fit(method='powell', max_iter=100)
    predict_tab = logit_fit.pred_table()
    train_error_rate = 1 - np.trace(predict_tab) / np.sum(predict_tab)
    y_predict_test = logit_fit.predict(test_df)
    y_predict_test = np.vectorize(int)(y_predict_test > 0.5)
    test_error_rate = 1 - np.sum(
        y_predict_test == test_df['y']) / len(y_predict_test)

    error_train.append(train_error_rate)
    error_test.append(test_error_rate)
    poly_degree.append(i_degree)

    # n-fold cross validation on training data
    for i_fold in range(n_folds):
        train_fold = train_df.loc[fold_ind != i_fold]
        test_fold = test_df.loc[fold_ind == i_fold]
        logit_model_fold = smf.logit(my_formula, data=train_fold)
        logit_fit_fold = logit_model_fold.fit(method='powell', max_iter=100)
        y_predict_fold = logit_fit_fold.predict(test_fold)
        y_predict_fold = np.vectorize(int)(y_predict_fold > 0.5)
        error = 1 - np.sum(y_predict_fold ==
                           test_fold['y']) / len(y_predict_fold)
        error_fold[i_degree].append(error)

    # Add next order to polynomial formula
    my_formula += ' + I(x1 ** ' + str(i_degree + 1) + ') + I(x2 ** ' +\
        str(i_degree + 1) + ')'

error_nfold_avg = []
for i in range(1, max_poly_degree + 1):
    error_nfold_avg.append(np.mean(error_fold[i]))

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.plot(poly_degree, error_train, c='b', label='Training')
ax1.plot(poly_degree, error_test, c='r', linestyle='--', label='Test')
ax1.plot(poly_degree, error_nfold_avg, c='k',
         linestyle=':', label='10-fold CV')
ax1.legend()
ax1.set_xlabel('Order of Polynomial')
ax1.set_ylabel('Error Rate')

# Estimate error rates with K Nearest Neighbors
error_test_knn = []
error_train_knn = []
k_neighbors = []
max_neighbors = 20
error_fold_knn = {}
for i_neighbors in range(1, max_neighbors + 1):
    error_fold_knn[i_neighbors] = []

for i_neighbors in range(1, max_neighbors + 1):
    knn = KNeighborsClassifier(n_neighbors=i_neighbors)
    knn.fit(train_df[['x1', 'x2']], train_df['y'])
    y_train_knn = knn.predict(train_df[['x1', 'x2']])
    y_test_knn = knn.predict(test_df[['x1', 'x2']])
    error_train_knn.append(1 - np.mean(y_train_knn == train_df['y']))
    error_test_knn.append(1 - np.mean(y_test_knn == test_df['y']))
    k_neighbors.append(i_neighbors)

    # n-fold cross validation on training data
    for i_fold in range(n_folds):
        train_fold = train_df.loc[fold_ind != i_fold]
        test_fold = train_df.loc[fold_ind == i_fold]
        knn_fold = KNeighborsClassifier(n_neighbors=i_neighbors)
        knn_fold.fit(train_fold[['x1', 'x2']], train_fold['y'])
        y_test_fold = knn_fold.predict(test_fold[['x1', 'x2']])
        error_fold_knn[i_neighbors].append(
            1 - np.mean(y_test_fold == test_fold['y']))

error_nfold_knn_avg = []
for i_neighbors in range(1, max_neighbors + 1):
    error_nfold_knn_avg.append(np.mean(error_fold_knn[i_neighbors]))

inv_k = [1/k for k in k_neighbors]
ax2 = fig.add_subplot(122)
ax2.plot(inv_k, error_train_knn, c='b', label='Training')
ax2.plot(inv_k, error_test_knn, c='r', linestyle='--', label='Test')
ax2.plot(inv_k, error_nfold_knn_avg, c='k', linestyle=':', label='10-fold CV')
ax2.set_xscale('log')
ax2.legend()
ax2.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
ax2.set_xticklabels([0.05, 0.1, 0.2, 0.5, 1.0])
ax2.set_xlabel(r'$\frac{1}{K}$')
ax2.set_ylabel('Error Rate')

fig.tight_layout()
