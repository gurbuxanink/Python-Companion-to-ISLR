# Text plot of Interpretability versus Flexibility
# Figure 2.7 in ISLR book
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# ax.plot([1,10], [1,10], marker='', linestyle='')
ax.text(0.05, 0.95, 'Subset Selection')
ax.text(0.05, 0.9, 'Lasso')
ax.text(0.2, 0.65, 'Least Squares')
ax.text(0.4, 0.4, 'Generalized Additive Models')
ax.text(0.4, 0.35, 'Trees')
ax.text(0.6, 0.1, 'Bagging, Boosting')
ax.text(0.6, 0.05, 'Support Vector Machines')

ax.set_xticks([0.1, 0.9])
ax.set_xticklabels(['Low', 'High'])
ax.set_xlabel('Flexibility')

ax.set_yticks([0.1, 0.9])
ax.set_yticklabels(['Low', 'High'])
ax.set_ylabel('Interpretability')
