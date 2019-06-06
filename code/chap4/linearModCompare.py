# Figure 4.10

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression as logit
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

knn1_model = KNN(n_neighbors=1)
knn5_model = KNN(n_neighbors=5)
lda_model = LDA()
qda_model = QDA()
logit_model = logit(solver='lbfgs')
my_models = {'knn1': knn1_model, 'knn5': knn5_model, 'lda': lda_model,
             'logit': logit_model, 'qda': qda_model}
error_rates = {'knn1': [], 'knn5': [], 'lda': [], 'logit': [], 'qda': []}

np.random.seed(911)
# Scenario 1
group1_means = [-0.7, -0.7]
group2_means = [0.7, 0.7]
test_size = 200
test1 = pd.DataFrame({
    'x1': np.random.normal(loc=group1_means[0], scale=1, size=test_size),
    'x2': np.random.normal(loc=group1_means[1], scale=1, size=test_size),
    'y': 0})
test2 = pd.DataFrame({
    'x1': np.random.normal(loc=group2_means[0], scale=1, size=test_size),
    'x2': np.random.normal(loc=group2_means[0], scale=1, size=test_size),
    'y': 1})
test_df1 = pd.concat((test1, test2), ignore_index=True)

sample_size = 20

for i in range(100):
    df1 = pd.DataFrame({
        'x1': np.random.normal(loc=group1_means[0], scale=1, size=sample_size),
        'x2': np.random.normal(loc=group1_means[1], scale=1, size=sample_size),
        'y': 0})

    df2 = pd.DataFrame(
        {'x1': np.random.normal(loc=group2_means[0], scale=1, size=sample_size),
         'x2': np.random.normal(loc=group2_means[1], scale=1, size=sample_size),
         'y': 1})

    my_df1 = pd.concat((df1, df2), ignore_index=True)

    for model_name in my_models.keys():
        model = my_models[model_name]
        fit = model.fit(my_df1[['x1', 'x2']], my_df1['y'])
        y_pred = fit.predict(test_df1[['x1', 'x2']])
        confusion = pd.crosstab(y_pred, test_df1['y'])
        error = (confusion.iloc[0, 1] + confusion.iloc[1, 0])/y_pred.size
        error_rates[model_name].append(error)

error_rates_df1 = pd.DataFrame(error_rates)

fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_subplot(131)
error_rates_df1.boxplot(grid=False, ax=ax1)
ax1.set_title('Scenario 1')


# Scenario 2
error_rates = {'knn1': [], 'knn5': [], 'lda': [], 'logit': [], 'qda': []}

group1_means = [-0.7, -0.7]
group2_means = [0.7, 0.7]
group1_cor = -0.5
group2_cor = -0.5
test_size = 200
X1 = np.random.multivariate_normal(group1_means,
                                   [[1, group1_cor], [group1_cor, 1]],
                                   size=test_size)
y = np.zeros(test_size, dtype=int)

X2 = np.random.multivariate_normal(group2_means,
                                   [[1, group2_cor], [group2_cor, 1]],
                                   size=test_size)
X = np.row_stack((X1, X2))
y = np.append(y, np.ones(test_size, dtype=int))

for i in range(100):
    X1_train = np.random.multivariate_normal(group1_means,
                                             [[1, group1_cor], [group1_cor, 1]],
                                             size=sample_size)
    y_train = np.zeros(sample_size, dtype=int)
    X2_train = np.random.multivariate_normal(group2_means,
                                             [[1, group2_cor], [group2_cor, 1]],
                                             size=sample_size)
    X_train = np.row_stack((X1_train, X2_train))
    y_train = np.append(y_train, np.ones(sample_size, dtype=int))

    for model_name in my_models.keys():
        model = my_models[model_name]
        fit = model.fit(X_train, y_train)
        y_pred = fit.predict(X)
        confusion = pd.crosstab(y_pred, y)
        error = (confusion.iloc[0, 1] + confusion.iloc[1, 0])/y_pred.size
        error_rates[model_name].append(error)

error_rates_df2 = pd.DataFrame(error_rates)
ax2 = fig.add_subplot(132)
error_rates_df2.boxplot(grid=False, ax=ax2)
ax2.set_title('Scenario 2')

# Scenario 3
error_rates = {'knn1': [], 'knn5': [], 'lda': [], 'logit': [], 'qda': []}

X1 = np.random.standard_t(df=2, size=(test_size, 2))
X2 = np.random.standard_t(df=2, size=(test_size, 2))
for i in range(2):
    X1[:, i] = X1[:, i] + group1_means[i]
    X2[:, i] = X2[:, i] + group2_means[i]

X = np.row_stack((X1, X2))
y = np.zeros(test_size, dtype=int)
y = np.append(y, np.ones(test_size, dtype=int))

sample_size = 50
for i in range(100):
    X1_train = np.random.standard_t(df=2, size=(sample_size, 2))
    X2_train = np.random.standard_t(df=2, size=(sample_size, 2))
    for i in range(2):
        X1_train[:, i] = X1_train[:, i] + group1_means[i]
        X2_train[:, i] = X2_train[:, i] + group2_means[i]
    X_train = np.row_stack((X1_train, X2_train))
    y_train = np.zeros(sample_size, dtype=int)
    y_train = np.append(y_train, np.ones(sample_size, dtype=int))

    for model_name in my_models.keys():
        model = my_models[model_name]
        fit = model.fit(X_train, y_train)
        y_pred = fit.predict(X)
        confusion = pd.crosstab(y_pred, y)
        error = (confusion.iloc[0, 1] + confusion.iloc[1, 0])/y_pred.size
        error_rates[model_name].append(error)

error_rates_df3 = pd.DataFrame(error_rates)
ax3 = fig.add_subplot(133)
error_rates_df3.boxplot(grid=False, ax=ax3)
ax3.set_title('Scenario 3')

fig.tight_layout()
