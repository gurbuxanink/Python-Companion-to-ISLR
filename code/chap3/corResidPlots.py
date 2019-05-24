# Plot figure 3.10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tsa.arima_process import ArmaProcess

cor_list = [0, 0.5, 0.9]
sample_size = 100
x = np.linspace(0, 10, sample_size)

fig = plt.figure(figsize=(6, 7))

for i in range(len(cor_list)):
    # pdb.set_trace()
    rho = cor_list[i]
    ar = np.array([1, -rho])
    ma = np.array([1])
    ar_object = ArmaProcess(ar, ma)
    sim_error = ar_object.generate_sample(nsample=sample_size)
    y = x + sim_error
    reg_model = smf.ols(formula='y ~ x', data=pd.DataFrame(dict(x=x, y=y)))
    reg_fit = reg_model.fit()

    subplot_num = 310 + i + 1
    ax = fig.add_subplot(subplot_num)
    # ax.plot(sim_error, 'r:')
    ax.axhline(y=0, linestyle='--')
    ax.plot(reg_fit.resid, 'o-', markersize=5, alpha=0.7)
    ax.set_ylabel('Residual')
    ax.set_title(r'$\rho = $' + str(rho))
    if i==2:
        ax.set_xlabel('Observation')

fig.tight_layout()
