import matplotlib.pyplot as plt
import numpy as np

separate_data = np.random.normal(loc=6, scale=1, size=150)
separate_data = np.reshape(separate_data, newshape=(75, 2))
separate_data[:25, 0] = separate_data[:25, 0] - 3
separate_data[:25, 1] = separate_data[:25, 1] + 3
separate_data[25:50, 0] = separate_data[25:50, 0] + 3
separate_data[50:, 0] = separate_data[50:, 0] - 3
separate_data[50:, 0] = separate_data[50:, 0] - 3

plot_char = ['+', 'x', 'o']
col_char = ['b', 'g', 'r']

fig = plt.figure()
ax1 = fig.add_subplot(121)

for i in [0, 1, 2]:
    ax1.scatter(separate_data[i * 25:(i + 1) * 25, 0],
                separate_data[i * 25:(i + 1) * 25, 1],
                marker=plot_char[i], c=col_char[i], alpha=0.7)

ax1.set_xlabel(r'$X_1$')
ax1.set_ylabel(r'$X_2$')

close_data = np.random.normal(loc=6, scale=1, size=150)
close_data = np.reshape(close_data, newshape=(75, 2))
close_data[:25, 0] = close_data[:25, 0] - 1.5
close_data[:25, 1] = close_data[:25, 1] + 1.5
close_data[25:50, 0] = close_data[25:50, 0] + 1.5
close_data[50:, 0] = close_data[50:, 0] - 1.5
close_data[50:, 0] = close_data[50:, 0] - 1.5

ax2 = fig.add_subplot(122)

for i in [0, 1, 2]:
    ax2.scatter(close_data[i * 25:(i + 1) * 25, 0],
                close_data[i * 25:(i + 1) * 25, 1],
                marker=plot_char[i], c=col_char[i], alpha=0.7)

ax2.set_xlabel(r'$X_1$')
ax2.set_ylabel(r'$X_2$')

fig.tight_layout()
