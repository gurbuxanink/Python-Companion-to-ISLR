# arrest_pca.py
# Code to create table 10.1, figures 10.1 and 10.3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.style.use('seaborn-whitegrid')

us_arrests = pd.read_csv('data/USArrests.csv', index_col=0)
us_arrests_normalized = (us_arrests - us_arrests.mean()) / us_arrests.std()


pca = PCA(n_components=2)
pca.fit(us_arrests_normalized)

pca_res = pd.DataFrame(
    pca.components_.T, index=us_arrests.columns, columns=['PC1', 'PC2'])

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot()
for crime in pca_res.index:
    pc1, pc2 = pca_res.loc[crime]
    ax.annotate(crime, xy=[0, 0], xytext=[pc1 * 3.5, pc2 * 3.5], color='red',
                arrowprops=dict(arrowstyle='<-', color='red', alpha=0.7),
                alpha=0.7, zorder=3)

us_arrests_transformed = pd.DataFrame(
    pca.transform(us_arrests_normalized), index=us_arrests.index,
    columns=['pc1', 'pc2'])

for state in us_arrests_transformed.index:
    x, y = us_arrests_transformed.loc[state]
    ax.text(x, y, state, horizontalalignment='center',
            verticalalignment='center', color='blue', alpha=0.7, zorder=1)

ax.set(xlim=[-3, 3], ylim=[-3, 3], xlabel='First Principal Component',
       ylabel='Second Principal Component')
ax.axis('equal')

# PCA without normalizing data
pca_raw = PCA(n_components=2)
pca_raw.fit(us_arrests)
pca_raw_res = pd.DataFrame(pca_raw.components_.T, index=us_arrests.columns,
                           columns=['PC1', 'PC2'])

arrests_transformed_raw = pd.DataFrame(
    pca_raw.transform(us_arrests), index=us_arrests.index,
    columns=['pc1', 'pc2'])

fig_raw = plt.figure(figsize=(8, 4))
ax1 = fig_raw.add_subplot(1, 2, 1)
for crime in pca_res.index:
    pc1, pc2 = pca_res.loc[crime]
    ax1.annotate(crime, xy=[0, 0], xytext=[pc1 * 2.5, pc2 * 2.5],
                 arrowprops=dict(arrowstyle='<-', color='red', alpha=0.7),
                 color='red', alpha=0.7, zorder=3)
us_arrests_transformed.plot(x='pc1', y='pc2', kind='scatter', marker='.',
                            alpha=0.7, ax=ax1)
ax1.set(title='Scaled')

ax2 = fig_raw.add_subplot(1, 2, 2)
for crime in pca_raw_res.index:
    pc1, pc2 = pca_raw_res.loc[crime]
    ax2.annotate(crime, xy=[0, 0], xytext=[pc1 * 100, pc2 * 100],
                 arrowprops=dict(arrowstyle='<-', color='red', alpha=0.7),
                 color='red', alpha=0.7, zorder=3)
arrests_transformed_raw.plot(x='pc1', y='pc2', kind='scatter', marker='.',
                             alpha=0.7, ax=ax2)
ax2.set(title='Unscaled')

for axi in fig_raw.axes:
    axi.set(xlabel='First Component', ylabel='Second Component')
    axi.axis('equal')
fig_raw.tight_layout()
