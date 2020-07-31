# roc_heart.py
# Code to plot figures 9.10 and 9.11
# ROC curves for Heart data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
plt.style.use('seaborn-whitegrid')

heart = pd.read_csv('data/heart.csv', index_col=0)
heart.dropna(inplace=True)

X_num = heart[['Age', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng',
               'Oldpeak', 'Slope', 'Ca']]
X_cat = heart[['Sex', 'ChestPain', 'Thal']]
X_dummies = pd.get_dummies(X_cat, drop_first=True)
X = pd.concat([X_num, X_dummies], axis=1)

y_dummies = pd.get_dummies(heart['AHD'], drop_first=True)
y = np.array(y_dummies).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)

svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)


def pos_rate(prob, model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1] > prob
    y_pred = y_pred.astype('int')
    true_pos_pred = np.sum((y_pred == 1) & (y_test == 1))
    true_pos = np.sum(y_test == 1)
    false_pos_pred = np.sum((y_pred == 1) & (y_test == 0))
    true_neg = np.sum(y_test == 0)
    return {'false_pos_rate': false_pos_pred / true_neg,
            'true_pos_rate': true_pos_pred / true_pos}


my_probs = np.linspace(0, 1, 21)
svc_roc_train = pd.DataFrame(
    [pos_rate(prob, svc, X_train, y_train) for prob in my_probs])
svc_roc_test = pd.DataFrame(
    [pos_rate(prob, svc, X_test, y_test) for prob in my_probs])
lda_roc_train = pd.DataFrame(
    [pos_rate(prob, lda, X_train, y_train) for prob in my_probs])
lda_roc_test = pd.DataFrame(
    [pos_rate(prob, lda, X_test, y_test) for prob in my_probs])

gamma_vals = [0.005, 0.001]
svm_rocs_train = {}
svm_rocs_test = {}
for gamma in gamma_vals:
    svm = SVC(kernel='rbf', gamma=gamma, probability=True)
    svm.fit(X_train, y_train)
    roc_train = pd.DataFrame(
        [pos_rate(prob, svm, X_train, y_train) for prob in my_probs])
    roc_test = pd.DataFrame(
        [pos_rate(prob, svm, X_test, y_test) for prob in my_probs])
    svm_rocs_train['gamma_' + str(gamma)] = roc_train
    svm_rocs_test['gamma_' + str(gamma)] = roc_test


def roc_plot(lda_roc_df, svc_roc_df, svm_rocs):
    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    lda_roc_df.plot(x='false_pos_rate', y='true_pos_rate', drawstyle='steps',
                    label='Linear Discriminant Analysis', linestyle='--',
                    ax=ax1)
    svc_roc_df.plot(x='false_pos_rate', y='true_pos_rate', drawstyle='steps',
                    label='Support Vector Classifier', ax=ax1)

    ax2 = fig.add_subplot(1, 2, 2)
    svc_roc_df.plot(x='false_pos_rate', y='true_pos_rate', drawstyle='steps',
                    label='Support Vector Classifier', linestyle='--', ax=ax2)
    for gamma_key in svm_rocs:
        gamma = gamma_key.split('_')[-1]
        svm_rocs[gamma_key].plot(
            x='false_pos_rate', y='true_pos_rate', drawstyle='steps',
            label=r'$SVM: \gamma = ' + gamma + '$', ax=ax2)

    for ax in fig.axes:
        ax.set(xlabel='False positive rate', ylabel='True positive rate')
        ax.plot([0, 1], [0, 1], linestyle=':', color='gray', alpha=0.7)
    fig.tight_layout()
    return fig


# Use grid search to find optimal values of C and gamma
# To get a high score, C must be very large and gamma must be very small
# svm = SVC()
# param_grid = {'C': [1000, 1e4, 1e5, 1e6], 'gamma': [1e-7, 1e-6, 1e-5]}
# grid = GridSearchCV(svm, param_grid=param_grid)
# grid.fit(X_train, y_train)
