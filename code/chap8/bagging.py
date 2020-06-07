# bagging.py
# Plot figures 8.8 and 8.9

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

Heart = pd.read_csv('data/Heart.csv', index_col=0)
Heart.dropna(inplace=True)

# Create dummy variables, then prepare X and y
X_numeric = Heart.drop(columns=['ChestPain', 'Thal', 'AHD'])
X_cat = pd.get_dummies(Heart[['ChestPain', 'Thal']], drop_first=False)
X = pd.concat([X_numeric, X_cat], axis=1)
y = Heart['AHD']

# Figure 8.8

# Figure 8.9
forest = RandomForestClassifier()
forest.fit(X, y)

feature_import = pd.Series(forest.feature_importances_, index=X.columns)
feature_import.sort_values(ascending=False, inplace=True)
feature_import = feature_import / feature_import.max() * 100

feature_import_fig, ax = plt.subplots()
feature_import.plot(kind='barh', color='red', alpha=0.9, ax=ax)
ax.set(xlabel='Variable Importance')
feature_import_fig.tight_layout()
