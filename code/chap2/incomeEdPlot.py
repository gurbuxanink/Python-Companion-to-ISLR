
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

income_ed = pd.read_csv('data/Income1.csv', index_col=0)


# The book does not provide the true function of Income versus Education
# This function is similar to the plot shown in the book
def edIncome(ed, a, b, c, d):
    return d + c * np.exp(a * ed + b) / (1 + np.exp(a * ed + b))


income_true = np.vectorize(edIncome)(income_ed['Education'], 0.6, -9.6, 60, 20)

ed_exog = np.linspace(income_ed['Education'].min(),
                      income_ed['Education'].max())

fig = plt.figure()
ax1 = fig.add_subplot(121)
income_ed.plot(x='Education', y='Income', kind='scatter', legend=False, ax=ax1,
               c='r', alpha=0.7)
ax1.set_xlabel('Years of Education')

ax2 = fig.add_subplot(122)
income_ed.plot(x='Education', y='Income', kind='scatter', legend=False, ax=ax2,
               c='r', alpha=0.7)
ax2.set_ylabel('')
ax2.set_xlabel('Years of Education')
ax2.plot(ed_exog, np.vectorize(edIncome)(ed_exog, 0.6, -9.6, 60, 20), 'b-')
ax2.vlines(income_ed['Education'], income_true, income_ed['Income'],
           colors='b', alpha=0.7)

fig.tight_layout()
