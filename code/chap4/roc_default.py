from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from statsmodels import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

default = datasets.get_rdataset('Default', 'ISLR').data
default['default_cat'] = np.vectorize(lambda x: int(x == 'Yes'))(
    default['default'])
default['student_cat'] = np.vectorize(lambda x: int(x == 'Yes'))(
    default['student'])

clf = LDA()
clf.fit(default[['student_cat', 'balance', 'income']], default['default_cat'])

default['prob_default'] = clf.predict_proba(default[['student_cat', 'balance',
                                                     'income']])[:, 1]

prob_def_thresholds = np.linspace(0, 1, 51)
false_pos_rate = []
true_pos_rate = []

for prob in prob_def_thresholds:
    default['default_predict'] = np.vectorize(
        lambda x: 'Yes' if x > prob else 'No')(default['prob_default'])
    confusion_df = pd.pivot_table(default[['default', 'default_predict']],
                                  index='default_predict',
                                  columns='default', aggfunc=len,
                                  margins=True)
    if 'Yes' in confusion_df.index:
        true_pos = confusion_df.loc['Yes', 'Yes'] / \
            confusion_df.loc['All', 'Yes']
        false_pos = confusion_df.loc['Yes', 'No'] / \
            confusion_df.loc['All', 'No']
    else:
        true_pos = (confusion_df.loc['All', 'Yes'] -
                    confusion_df.loc['No', 'Yes']) / confusion_df.loc['All', 'Yes']
        false_pos = (confusion_df.loc['All', 'No'] -
                     confusion_df.loc['No', 'No']) / confusion_df.loc['All', 'No']

    false_pos_rate.append(false_pos)
    true_pos_rate.append(true_pos)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(false_pos_rate, true_pos_rate)
ax.plot([0, 1], [0, 1], linestyle=':')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
