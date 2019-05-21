import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.datasets as datasets

credit = datasets.get_rdataset('Credit', package='ISLR').data

axes  = pd.plotting.scatter_matrix(
    credit[['Balance', 'Age', 'Cards', 'Education', 'Income', 'Limit',
            'Rating']], alpha=0.6, s=5, figsize=(6,6))

[plt.setp(item.xaxis.get_label(), 'size', 7) for item in axes.ravel()]
[plt.setp(item.xaxis.get_majorticklabels(), 'size', 7)
 for item in axes.ravel()]
[plt.setp(item.yaxis.get_label(), 'size', 7) for item in axes.ravel()]
[plt.setp(item.yaxis.get_majorticklabels(), 'size', 7)
 for item in axes.ravel ()]
plt.tight_layout()
