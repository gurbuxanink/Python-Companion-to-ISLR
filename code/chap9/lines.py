# A class for straight line
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class line(object):
    '''A straight line
    Can be created using one of three methods: (i) from slope and intercept,
    (ii) from slope and a point on line, (ii) from two points on line.'''

    def __init__(self):
        self.slope = np.nan
        self.intercept = np.nan

    def from_slope_intercept(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def from_slope_point(self, slope, point1):
        self.slope = slope
        x1, y1 = point1[0], point1[1]
        self.intercept = y1 - slope * x1

    def from_two_points(self, point1, point2):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        self.slope = (y2 - y1) / (x2 - x1)
        self.intercept = y2 - self.slope * x2

    def __str__(self):
        '''Returns slope and intercept'''
        return 'slope: ' + str(self.slope) + ', intercept: ' + \
            str(self.intercept)

    @property
    def slope_(self):
        return self.slope

    @property
    def intercept_(self):
        return self.intercept

    def get_y(self, x):
        return self.slope * x + self.intercept

    def get_x(self, y):
        return (y - self.intercept) / self.slope

    def get_intersection(self, other_line):
        '''Return intersection point of two lines'''
        m1, b1 = self.slope, self.intercept
        m2, b2 = other_line.slope_, other_line.intercept_
        x = -(b1 - b2) / (m1 - m2)
        y = self.get_y(x)
        return [x, y]

    def plot_line(self, x_range, ax=None, **kwargs):
        '''x_range is tuple of (x_min, x_max)'''
        x_array = np.linspace(x_range[0], x_range[1])
        y_array = self.get_y(x_array)
        if ax is None:
            ax = plt.gca()
        ax.plot(x_array, y_array, **kwargs)
