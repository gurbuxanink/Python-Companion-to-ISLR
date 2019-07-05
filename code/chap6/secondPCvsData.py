# Figure 6.17
# Plot data versus second principal component
# We do not have actual data used in the book.  Therefore, we use
# 2-year and 10-year treasury yield history in 2018.

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

treasury = pd.read_csv('data/treasury.csv')
treasury['Date'] = treasury['Date'].apply(
    lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
treasury.set_index('Date')
treasury_diff = treasury.diff(periods=-1) * 100  # convert to basis points
treasury_2s10s = treasury_diff[['2 YR', '10 YR']].iloc[:-1]  # drop na
mean_2s10s = treasury_2s10s.mean()
treasury_2s10s = treasury_2s10s - mean_2s10s

eigval, eigvec = np.linalg.eig(np.cov(treasury_2s10s, rowvar=False))
pc1 = eigvec[:, 1]
pc2 = eigvec[:, 0]
treasury_2s10s_pc2 = np.dot(treasury_2s10s, pc2)

y_vars = ['10 YR', '2 YR']

fig = plt.figure(figsize=(8, 4))
for i in range(len(y_vars)):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.scatter(treasury_2s10s_pc2, treasury_2s10s[y_vars[i]], c='r',
               alpha=0.6, s=20)
    ax.set_xlabel('Second Principal Component')
    ax.set_ylabel(y_vars[i])

fig.tight_layout()
