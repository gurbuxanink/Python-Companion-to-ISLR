import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

auto_full = pd.read_csv('data/Auto.csv', na_values='?')
auto = auto_full.dropna()

linear_model = smf.ols(formula='mpg ~ horsepower', data=auto)
linear_fit = linear_model.fit()

deg2_model = smf.ols(formula='mpg ~ horsepower + I(horsepower**2)', data=auto)
deg2_fit = deg2_model.fit()

deg5_model = smf.ols(formula='mpg ~ horsepower + I(horsepower**2) + I(horsepower**3) + I(horsepower**4) + I(horsepower**5)', data=auto)
deg5_fit = deg5_model.fit()

# Easier to type than five terms in formula
polyfit_deg5 = np.polyfit(x=auto['horsepower'], y=auto['mpg'], deg=5)

hp_array = np.linspace(auto['horsepower'].min(), auto['horsepower'].max())

fig = plt.figure()
ax = fig.add_subplot(111)
auto.plot.scatter('horsepower', 'mpg', color='grey', alpha=0.6, ax=ax)
ax.plot(hp_array, linear_fit.predict(exog=dict(horsepower=hp_array)),
        color='brown', label='Linear')
ax.plot(hp_array, deg2_fit.predict(exog=dict(horsepower=hp_array)),
        c='b', linestyle='--', label='Degree 2')
ax.plot(hp_array, deg5_fit.predict(exog=dict(horsepower=hp_array)),
        c='g', linestyle='-.', label='Degree 5')
ax.legend()
ax.set_xlabel('Horsepower')
ax.set_ylabel('Miles per gallon')

# Overplots degree 5 fit
# ax.plot(hp_array, np.polyval(polyfit_deg5, hp_array), c='g', linestyle='-')
