# treePartition.py
# Plot figure 8.3
# How to get axis labels cloeser in 3d plot (ax4)?

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


fig = plt.figure()

# Partitions of two-dimensional space that could not result from
# recursive binary splitting
ax1 = fig.add_subplot(2, 2, 1)

line_endpoints = np.array(
    [[0, 0.6, 0.6, 0.6],
     [0, 0.7, 0.6, 0.7],
     [0.6, 0.6, 0.6, 0.7],
     [0.3, 0, 0.3, 0.6],
     [0.3, 0.3, 0.7, 0.3],
     [0.7, 0.3, 0.7, 0.65],
     [0.6, 0.65, 0.7, 0.65],
     [0.5, 0, 0.5, 0.3]])

for i in range(line_endpoints.shape[0]):
    ax1.plot(line_endpoints[i, 0::2], line_endpoints[i, 1::2],
             color='gray')

ax1.set(xlim=[0, 1], ylim=[0, 1], xticks=[], yticks=[],
        xlabel=r'$X_1$', ylabel=r'$X_2$')

# Output of recursive binary splitting of two-dimensional space
ax2 = fig.add_subplot(2, 2, 2)
t1, t2, t3, t4 = (0.4, 0.3, 0.6, 0.7)
line_segments = np.array(
    [[t1, 0, t1, 1],
     [t3, 0, t3, 1],
     [0, t2, t1, t2],
     [t3, t4, 1, t4]])

for i in range(line_segments.shape[0]):
    ax2.plot(line_segments[i, 0::2], line_segments[i, 1::2],
             color='gray')

region_names = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$']
region_name_locations = np.array(
    [[t1/2, t2/2], [t1/2, (1 + t2)/2], [(t1 + t3) / 2, 0.5],
     [(1 + t3)/2, t4/2], [(1 + t3)/2, (1 + t4)/2]])

for i in range(len(region_names)):
    ax2.text(region_name_locations[i, 0],
             region_name_locations[i, 1], region_names[i],
             ha='center', fontsize=10)

ax2.set(xlim=[0, 1], ylim=[0, 1], xticks=[t1, t3], yticks=[t2, t4],
        xticklabels=[r'$t_1$', r'$t_3$'], yticklabels=[r'$t_2$', r'$t_4$'],
        xlabel=r'$X_1$', ylabel=r'$X_2$')

# Tree corresponding to recursive binary splitting in ax2
ax3 = fig.add_subplot(2, 2, 3)

line_segments = np.array(
    [[4, 0.9, 4, 0.92],
     [2, 0.9, 6, 0.9],
     [2, 0.6, 2, 0.9],
     [6, 0.6, 6, 0.9],
     [1, 0.6, 3, 0.6],
     [5, 0.6, 7, 0.6],
     [1, 0.3, 1, 0.6],
     [3, 0.3, 3, 0.6],
     [5, 0.3, 5, 0.6],
     [7, 0.3, 7, 0.6],
     [6.5, 0.3, 7.5, 0.3],
     [6.5, 0, 6.5, 0.3],
     [7.5, 0, 7.5, 0.3]])

for i in range(line_segments.shape[0]):
    ax3.plot(line_segments[i, 0::2], line_segments[i, 1::2],
             color='gray')

label_locations = np.array([[4, 0.9], [2, 0.6], [6, 0.6], [7, 0.3]])
region_locations = np.array(
    [[1, 0.3], [3, 0.3], [5, 0.3], [6.5, 0], [7.5, 0]])
labels = [r'$X_1 < t_1$', r'$X_2 < t_2$', r'$X_1 < t_3$',
          r'$X_2 < t_4$']
region_names = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$']
for i in range(label_locations.shape[0]):
    ax3.text(label_locations[i, 0], label_locations[i, 1] + 0.02,
             labels[i], ha='right', fontsize=7)
for i in range(region_locations.shape[0]):
    ax3.text(region_locations[i, 0], region_locations[i, 1] - 0.08,
             region_names[i], ha='center', fontsize=8)
ax3.axis('off')

# Artificial data to create partitions in ax2
xx1 = np.linspace(0, 1)
xx2 = np.linspace(0, 1)
xx1_grid, xx2_grid = np.meshgrid(xx1, xx2)
X = np.vstack((xx1_grid.ravel(), xx2_grid.ravel())).T


# def map_region(x1, x2):
def map_region(x):
    x1, x2 = x[0], x[1]
    if x1 < t1:
        if x2 < t2:
            return 1
        else:
            return 2
    elif x1 < t3:
        return 3
    else:
        if x2 < t4:
            return 4
        else:
            return 5


y = np.apply_along_axis(map_region, 1, X)
y_grid = y.reshape(xx1_grid.shape)

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(xx1_grid, xx2_grid, y_grid,
                 cmap=plt.cm.get_cmap('RdBu', 5), alpha=0.7)
ax4.set(xticks=[], yticks=[], zticks=[],
        xlabel=r'$X_1$', ylabel=r'$X_2$', zlabel=r'$Y$')
ax4.xaxis.pane.fill = False
ax4.yaxis.pane.fill = False
ax4.zaxis.pane.fill = False

fig.tight_layout()
