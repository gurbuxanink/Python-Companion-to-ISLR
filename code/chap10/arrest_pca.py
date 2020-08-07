# arrest_pca.py
# Code to create table 10.1

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
