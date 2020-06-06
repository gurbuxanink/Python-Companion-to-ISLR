# treeVsLinModel.py
# Code to plot figure 8.7

import matplotlib.pyplot as plt
import numpy as np


x_min, x_max = -2.25, 2.25
y_min, y_max = x_min, x_max
x = np.linspace(x_min, x_max)
y = -0.5 + 0.4 * x


fig, axes = plt.subplots(2, 2)

ax = axes[0, 0]
ax.plot(x, y, color='black', alpha=0.7)
ax.fill_between(x, y1=y, y2=y_max, color='green', alpha=0.5)

ax = axes[0, 1]
# ax.plot(x, y)
ax.fill_between(x, y1=y, y2=y_max, color='green', alpha=0.5, zorder=1)
x1, x2 = np.quantile(x, [0.33, 0.66])
y1, y2, y3 = np.mean(y[:16]), np.mean(y[16:33]), np.mean(y[33:])
line_segments = np.array(
    [[x1, y_min, x1, y2],
     [x2, y2, x2, y3],
     [x_min, y1, x1, y1],
     [x_min, y2, x_max, y2],
     [x_min, y3, x_max, y3]])
for line in line_segments:
    ax.plot(line[::2], line[1::2], color='black', alpha=0.7, zorder=2)

ax = axes[1, 0]
ax.fill_between(x[:16], y1=y_min, y2=y_max, color='green',
                edgecolor=None, alpha=0.5, zorder=1)
ax.fill_between(x[15:], y1=x[32], y2=y_max, color='green',
                edgecolor=None, alpha=0.5, zorder=1)
ax.plot([x_min, x_max], [y_min, y_max], color='black', alpha=0.7, zorder=2)

ax = axes[1, 1]
ax.fill_between(x[:16], y1=y_min, y2=y_max, color='green',
                edgecolor=None, alpha=0.5, zorder=1)
ax.fill_between(x[15:], y1=x[32], y2=y_max, color='green',
                edgecolor=None, alpha=0.5, zorder=1)
ax.plot([x[15], x[15]], [y_min, y_max], color='black', alpha=0.7, zorder=2)
ax.plot([x[15], x_max], [x[32], x[32]], color='black', alpha=0.7, zorder=2)

for ax in axes.flatten():
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
           xlabel=r'$X_1$', ylabel=r'$X_2$')

fig.tight_layout()
