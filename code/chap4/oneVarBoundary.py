# Figure 4.4
# TODO calculate error rates with test data

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from scipy.stats import norm

sample_size = 20
np.random.seed(911)
x = np.random.normal(loc=-1.25, size=sample_size)
x = np.append(x, np.random.normal(loc=1.25, size=sample_size))
x = x[:, np.newaxis]
y = np.append(np.zeros(sample_size, dtype=int),
              np.zeros(sample_size, dtype=int) + 1)

gnb = GaussianNB()
gnb.fit(x, y)

clf = LDA()
clf.fit(x, y)

x_array = np.linspace(-6, 6, 500)
y_pred_bayes = gnb.predict(x_array[:, np.newaxis])
boundary_ind_bayes = np.min(np.where(y_pred_bayes == 1))
x_boundary_bayes = x_array[boundary_ind_bayes]

y_pred_lda = clf.predict(x_array[:, np.newaxis])
boundary_ind_lda = np.min(np.where(y_pred_lda == 1))
x_boundary_lda = x_array[boundary_ind_lda]

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.plot(x_array, norm.pdf(x_array, loc=-1.5), color='green')
ax1.plot(x_array, norm.pdf(x_array, loc=1.5), color='brown', linestyle='--')
ax1.axvline(x=x_boundary_bayes, linestyle='--', c='k')

ax2 = fig.add_subplot(122)
pcol = ['green', 'brown']
for i in range(2):
    ax2.hist(x[y == i], color=pcol[i], alpha=0.5)
ax2.axvline(x=x_boundary_bayes, linestyle='--', c='k')
ax2.axvline(x=x_boundary_lda, c='k')

fig.tight_layout()
