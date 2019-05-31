# Plot figure 4.1

from statsmodels import datasets
import matplotlib.pyplot as plt
import numpy as np

credit = datasets.get_rdataset('Default', 'ISLR', cache=True)
credit_data = credit.data
credit_sample = credit_data.iloc[np.random.choice(credit_data.shape[0], 1000,
                                                  replace=False)]

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
credit_sample.loc[credit_sample['default'] == 'No'].plot(x='balance', y='income',
                                                     alpha=0.5, kind='scatter',
                                                     ax=ax1)
credit_data.loc[credit_data['default'] == 'Yes'].plot(x='balance', y='income',
                                                      marker='+', color='brown',
                                                      kind='scatter', alpha=0.9,
                                                      ax=ax1)

ax1.set_xlabel('Balance')
ax1.set_ylabel('Income')

ax2 = fig.add_subplot(143)
credit_data.boxplot(column='balance', by='default', ax=ax2, widths=0.5,
                    grid=False)
ax2.set_yticks([0, 1000, 2000, 3000])
ax2.set_yticklabels([0, 1000, 2000, 3000], rotation=90)
ax2.set_title('Balance')
ax2.set_xlabel('Default')

ax3 = fig.add_subplot(144)
credit_data.boxplot(column='income', by='default', ax=ax3, widths=0.5,
                    grid=False)
ax3.set_yticks([0, 25000, 50000, 75000])
ax3.set_yticklabels([0, 25000, 50000, 75000], rotation=90)
ax3.set_xlabel('Default')
ax3.set_title('Income')

fig.suptitle('')
fig.tight_layout()
