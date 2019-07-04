# Figure 6.14
# We do not have actual data set used in the book
# We use 2-year and 10-year treasury yield history

import matplotlib.pyplot as plt
# from statsmodels import datasets
import numpy as np
from datetime import datetime
import pandas as pd

treasury = pd.read_csv('data/treasury.csv')
treasury['Date'] = treasury['Date'].apply(
    lambda d: datetime.strptime(d, '%Y-%m-%d'))
treasury.set_index('Date', inplace=True)
treasury_diff = treasury.diff(periods=-1) * 100  # basis points conversion
treasury_2s10s = treasury_diff[['2 YR', '10 YR']].iloc[:-1]  # drop na
mean_2s10s = treasury_2s10s.mean()
# treasury_2s10s = treasury_2s10s - mean_2s10s

eigval, eigvec = np.linalg.eig(np.cov(treasury_2s10s, rowvar=False))
pc1 = eigvec[:, 1]
pc2 = eigvec[:, 0]
pc1_2s10s = np.dot(treasury_2s10s, pc1)
recover_2s10s_pc1 = np.dot(pc1_2s10s[:, np.newaxis], pc1[:, np.newaxis].T)
pc2_2s10s = np.dot(treasury_2s10s, pc2)
recover_2s10s_pc2 = np.dot(pc2_2s10s[:, np.newaxis], pc2[:, np.newaxis].T)
# recover_2s10s_all = recover_2s10s_pc1 + recover_2s10s_pc2


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

treasury_diff.plot(x='10 YR', y='2 YR', kind='scatter',
                   c='r', ax=ax, alpha=0.7)
ax.plot(recover_2s10s_pc1[:, 0], recover_2s10s_pc1[:, 1], c='g', label='pc1',
        linewidth=1)
ax.plot(recover_2s10s_pc2[:, 0], recover_2s10s_pc2[:, 1], c='b',
        linestyle=':', label='pc2')
# ax.scatter(recover_2s10s_all[:, 1], recover_2s10s_all[:, 0], c='r', marker='x',
#            alpha=0.7)
ax.legend()
