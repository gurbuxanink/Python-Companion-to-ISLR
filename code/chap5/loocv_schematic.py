# Figure 5.3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Circle

rect1 = Rectangle((0.02, 0.9), width=0.96, height=0.1, fill=False,
                  edgecolor='k')
arrow = Arrow(0.5, 0.88, 0, -0.26, width=0.3)
rect2 = Rectangle((0.02, 0.5), width=0.96, height=0.09, color='b', alpha=0.2)
rect3 = Rectangle((0.02, 0.4), width=0.96, height=0.09, color='b', alpha=0.2)
rect4 = Rectangle((0.02, 0.3), width=0.96, height=0.09, color='b', alpha=0.2)
rect5 = Rectangle((0.02, 0), width=0.96, height=0.09, color='b', alpha=0.2)

fig = plt.figure()
ax = fig.add_subplot(111, frameon=False)
for shape in (rect1, rect2, rect3, rect4, rect5, arrow):
    ax.add_artist(shape)

for dot_height in [0.15, 0.2, 0.25]:
    ax.add_artist(Circle((0.5, dot_height), radius=0.007))

txt_height = [0.53, 0.43, 0.33]
text_place = np.arange(0.05, 0.2, 0.05)

for i in range(3):
    ax.text(text_place[i], 0.93, str(i+1))
    ax.text(text_place[i], 0.03, str(i+1))

ax.text(0.9, 0.93, 'n')
ax.text(0.9, 0.03, 'n', bbox=dict(facecolor='red', alpha=0.4))

for box in range(1, 4):
    y = txt_height[box - 1]
    for i in range(3):
        if (i + 1) == box:
            ax.text(text_place[i], y, str(i + 1),
                    bbox=dict(facecolor='red', alpha=0.4))
        else:
            ax.text(text_place[i], y, str(i + 1))
    ax.text(0.9, y, 'n')

ax.set_xticks([])
ax.set_yticks([])
