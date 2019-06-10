# Figure 5.5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

rect1 = Rectangle((0.02, 0.9), width=0.96, height=0.1, fill=False,
                  edgecolor='k')
arrow = Arrow(0.5, 0.88, 0, -0.38, width=0.2)

rect_y = [0, 0.1, 0.2, 0.3, 0.4]
rect_x = [0.8, 0.6, 0.4, 0.2, 0.02]

fig = plt.figure()
ax = fig.add_subplot(111, frameon=False)
ax.add_artist(rect1)
ax.text(0.05, 0.93, '1 2 3')
ax.text(0.9, 0.93, 'n')
ax.add_artist(arrow)

for i in range(5):
    ax.add_artist(Rectangle((0.02, rect_y[i]), width=0.96, height=0.09,
                            color='b', alpha=0.2))
    ax.add_artist(Rectangle((rect_x[i], rect_y[i]), width=0.18, height=0.09,
                            color='r', alpha=0.4))
    ax.text(0.05, rect_y[i] + 0.03, '11 76 5')
    ax.text(0.9, rect_y[i] + 0.03, '47')

ax.set_xticks([])
ax.set_yticks([])
