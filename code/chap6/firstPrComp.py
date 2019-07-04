# Figure 6.15
# Plot first principal component and its distance from data points
# We do not have actual data used in the book.  Therefore, we use
# 2-year and 10-year treasury yield history in 2018.

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd


treasury = pd.read_csv('data/treasury.csv')
treasury['Date'] = treasury['Date'].apply(
    lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
treasury.set_index('Date', inplace=True)
treasury_diff = treasury.diff(periods=-1) * 100  # convert to basis points
treasury_2s10s = treasury_diff[['2 YR', '10 YR']].iloc[:-1]  # drop na
mean_2s10s = treasury_2s10s.mean()
treasury_2s10s = treasury_2s10s - mean_2s10s

eigval, eigvec = np.linalg.eig(np.cov(treasury_2s10s, rowvar=False))
pc1 = eigvec[:, 1]
pc2 = eigvec[:, 0]
treasury_2s10s_pc1 = np.dot(treasury_2s10s, pc1)
treasury_2s10s_pc2 = np.dot(treasury_2s10s, pc2)
recover_2s10s_pc1 = np.dot(
    treasury_2s10s_pc1[:, np.newaxis], pc1[:, np.newaxis].T)

recover_2s10s_pc2 = np.dot(
    treasury_2s10s_pc2[:, np.newaxis], pc2[:, np.newaxis].T)

recover_2s10s_all = recover_2s10s_pc1 + recover_2s10s_pc2 + \
    np.array(mean_2s10s)[np.newaxis, :]

np.random.seed(911)
index = np.random.choice(treasury_2s10s.shape[0], size=50)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
treasury_2s10s.iloc[index].plot(x='10 YR', y='2 YR', kind='scatter', c='r',
                                ax=ax1, alpha=0.7)
ax1.plot(recover_2s10s_pc1[:, 1], recover_2s10s_pc1[:, 0], linewidth=1, c='b',
         alpha=0.7)
# ax1.scatter(recover_2s10s_pc1[index, 1], recover_2s10s_pc1[index, 0],
#             marker='x', c='k')
for i in index:
    ax1.plot([recover_2s10s_pc1[i, 1], treasury_2s10s.iloc[i, 1]],
             [recover_2s10s_pc1[i, 0], treasury_2s10s.iloc[i, 0]],
             linestyle='--', c='k', alpha=0.6)
ax1.set_xlim(treasury_2s10s.min()[1], treasury_2s10s.max()[1])
ax1.set_ylim(treasury_2s10s.min()[0], treasury_2s10s.max()[0])

ax2 = fig.add_subplot(122)
for i in index:
    ax2.plot([treasury_2s10s_pc1[i], treasury_2s10s_pc1[i]],
             [0, treasury_2s10s_pc2[i]], linestyle='--', c='k', alpha=0.6)
    ax2.scatter(treasury_2s10s_pc1[i], treasury_2s10s_pc2[i], c='r',
                s=20, alpha=0.7)
ax2.axhline(y=treasury_2s10s_pc2.mean(), c='b', alpha=0.7)
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_xlabel('First Principal Component')
ax2.set_ylabel('Second Principal Component')

fig.tight_layout()
