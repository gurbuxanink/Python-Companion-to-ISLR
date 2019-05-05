# Boxplots of changes in S&P 500 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# smarket = pd.read_csv('data/Smarket.csv', index_col=0)
smarket = sm.datasets.get_rdataset('Smarket', 'ISLR').data

fig = plt.figure()
ax1 = fig.add_subplot(131)
smarket.boxplot(column='Lag1', by='Direction', ax=ax1, grid=False, widths=0.6)
ax1.set_xlabel("Today's direction")
ax1.set_ylabel('Percentage change in S&P 500')
ax1.set_title('Yesterday')

ax2 = fig.add_subplot(132)
smarket.boxplot(column='Lag2', by='Direction', ax=ax2, grid=False, widths=0.6)
ax2.set_xlabel("Today's direction")
ax2.set_title('Two days previous')

ax3 = fig.add_subplot(133)
smarket.boxplot(column='Lag3', by='Direction', ax=ax3, grid=False, widths=0.6)
ax3.set_xlabel("Today's direction")
ax3.set_title('Three days previous')

fig.suptitle('')
fig.tight_layout()
