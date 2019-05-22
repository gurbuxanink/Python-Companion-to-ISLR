---
header-includes:
- '\usepackage{amsmath,booktabs,placeins}'
- '\hypersetup{colorlinks=true, allcolors=blue, linkbordercolor=white}'
title: A Python Companion to ISLR
---

\usepackage{amsmath,booktabs,placeins}

\hypersetup{colorlinks=true, allcolors=blue, linkbordercolor=white}

Introduction
============

Figure [fig:introFig1](fig:introFig1) shows graphs of Wage versus three
variables.

![`Wage` data, which contains income survey information for males from
the central Atlantic region of the United States. Left: `wage` as a
function of `age`. On average, `wage` increases with `age` until about
60 years of age, at which point it begins to decline. Center: `wage` as
a function of `year`. There is a slow but steady increase of
approximately \$10,000 in the average `wage` between 2003 and 2009.
Right: Boxplots displaying `wage` as a function of `education`, with 1
indicating the lowest level (no highschool diploma) and 5 the highest
level (an advanced graduate degree). On average, `wage` increases with
the level of `education`.](figures/fig1_1.png "introFig1")

Figure [fig:introFig2](fig:introFig2) shows boxplots of previous days\'
percentage changes in S&P 500 grouped according to today\'s change `Up`
or `Down`.

![Left: Boxplots of the previous day\'s percentage change in the S&P 500
index for the days for which the market increased or decreased, obtained
from the `Smarket` data. Center and Right: Same as left panel, but the
percentage changes for two and three days previous are
shown.](figures/fig1_2.png "introFig2")

\FloatBarrier

Statistical Learning
====================

What is Statistical Learning?
-----------------------------

Figure [fig:statLearnFig1](fig:statLearnFig1) shows scatter plots of
`sales` versus `TV`, `radio`, and `newspaper` advertising. In each
panel, the figure also includes an OLS regression line.

![The `Advertising` data set. The plot displays `sales`, in thousands of
units, as a function of `TV`, `radio`, and `newspaper` budgets, in
thousands of dollars, for 200 different markets. In each plot we show
the simple least squares fit of `sales` to that variable. In other
words, each red line represents a simple model that can be used to
predict `sales` using `TV`, `radio`, and `newspaper`,
respectively.](figures/fig2_1.png "statLearnFig1")

Figure [fig:statLearnFig2](fig:statLearnFig2) is a plot of `Income`
versus `Years of Education` from the Income data set. In the left panel,
the \`\`true\'\' function (given by blue line) is actually my guess.

![The `Income` data set. Left: The red dots are the observed values of
`income` (in tens of thousands of dollars) and `years of education` for
30 individuals. Right: The blue curve represents the true underlying
relationship between `income` and `years of education`, which is
generally unknown (but is known in this case because the data are
simulated). The vertical lines represent the error associated with each
observation. Note that some of the errors are positive (when an
observation lies above the blue curve) and some are negative (when an
observation lies below the curve). Overall, these errors have
approximately mean zero.](figures/fig2_2.png "statLearnFig2")

Figure [fig:statLearnFig3](fig:statLearnFig3) is a plot of `Income`
versus `Years of Education` and `Seniority` from the `Income` data set.
Since the book does not provide the true values of `Income`,
\`\`true\'\' values shown in the plot are actually third order
polynomial fit.

![The plot displays `income` as a function of `years of education` and
`seniority` in the `Income` data set. The blue surface represents the
true underlying relationship between `income` and `years of education`
and `seniority`, which is known since the data are simulated. The red
dots indicate the observed values of these quantities for 30
individuals.](figures/fig2_3.png "statLearnFig3")

Figure [fig:statLearnFig4](fig:statLearnFig4) shows an example of the
parametric approach applied to the `Income` data from previous figure.

![A linear model fit by least squares to the `Income` data from figure
[fig:statLearnFig3](fig:statLearnFig3). The observations are shown in
red, and the blue plane indicates the least squares fit to the
data.](figures/fig2_4.png "statLearnFig4")

Figure [fig:statLearnFig7](fig:statLearnFig7) provides an illustration
of the trade-off between flexibility and interpretability for some of
the methods covered in this book.

![A representation of the tradeoff between flexibility and
interpretability, using different statistical learning methods. In
general, as the flexibility of a method increases, its interpretability
decreases.](figures/figure2_7.png "statLearnFig7")

Figure [fig:statLearnFig8](fig:statLearnFig8) provides a simple
illustration of the clustering problem.

![A clustering data set involving three groups. Each group is shown
using a different colored symbol. Left: The three groups are
well-separated. In this setting, a clustering approach should
successfully identify the three groups. Right: There is some overlap
among the groups. Now the clustering taks is more
challenging.](figures/fig2_8.png "statLearnFig8")

Assessing Model Accuracy
------------------------

Figure [fig:statLearnFig9](fig:statLearnFig9) illustrates the tradeoff
between training MSE and test MSE. We select a \`\`true function\'\'
whose shape is similar to that shown in the book. In the left panel, the
orange, blue, and green curves illustrate three possible estimates for
$f$ given by the black curve. The orange line is the linear regression
fit, which is relatively inflexible. The blue and green curves were
produced using *smoothing splines* from `UnivariateSpline` function in
`scipy` package. We obtain different levels of flexibility by varying
the parameter `s`, which affects the number of knots.

For the right panel, we have chosen polynomial fits. The degree of
polynomial represents the level of flexibility. This is because the
function `UnivariateSpline` does not more than five degrees of freedom.

When we repeat the simulations for figure
[fig:statLearnFig9](fig:statLearnFig9), we see considerable variation in
the right panel MSE plots. But the overall conclusion remains the same.

![Left: Data simulated from $f$, shown in black. Three estimates of $f$
are shown: the linear regression line (orange curve), and two smoothing
spline fits (blue and green curves). Right: Training MSE (grey curve),
test MSE (red curve), and minimum possible test MSE over all methods
(dashed grey line).](figures/fig2_9.png "statLearnFig9")

Figure [fig:statLearnFig10](fig:statLearnFig10) provides another example
in which the true $f$ is approximately linear.

![Details are as in figure [fig:statLearnFig9](fig:statLearnFig9) using
a different true $f$ that is much closer to linear. In this setting,
linear regression provides a very good fit to the
data.](figures/fig2_10.png "statLearnFig10")

Figure [fig:statLearnFig11](fig:statLearnFig11) displays an example in
which $f$ is highly non-linear. The training and test MSE curves still
exhibit the same general patterns.

![Details are as in figure [fig:statLearnFig9](fig:statLearnFig9), using
a different $f$ that is far from linear. In this setting, linear
regression provides a very poor fit to the
data.](figures/fig2_11.png "statLearnFig11")

Figure [fig:statLearnFig12](fig:statLearnFig12) displays the
relationship between bias, variance, and test MSE. This relationship is
referred to as *bias-variance trade-off*. When simulations are repeated,
we see considerable variation in different graphs, especially for MSE
lines. But overall shape remains the same.

![Squared bias (blue curve), variance (orange curve), $Var(\epsilon)$
(dashed line), and test MSE (red curve) for the three data sets in
figures [fig:statLearnFig9](fig:statLearnFig9) -
[fig:statLearnFig11](fig:statLearnFig11). The vertical dotted line
indicates the flexibility level corresponding to the smallest test
MSE.](figures/fig2_12.png "statLearnFig12")

Figure [fig:statLearnFig13](fig:statLearnFig13) provides an example
using a simulated data set in two-dimensional space consisting of
predictors $X_1$ and $X_2$.

![A simulated data set consisting of 200 observations in two groups,
indicated in blue and orange. The dashed line represents the Bayes
decision boundary. The orange background grid indicates the region in
which a test observation will be assigned to the orange class, and blue
background grid indicates the region in which a test observation will be
assigned to the blue class.](figures/fig2_13.png "statLearnFig13")

Figure [fig:statLearnFig15](fig:statLearnFig15) displays the KNN
decision boundary, using $K=10$, when applied to the simulated data set
from figure [fig:statLearnFig13](fig:statLearnFig13). Even though the
true distribution is not known by the KNN classifier, the KNN decision
making boundary is very close to that of the Bayes classifier.

![The firm line indicates the KNN decision boundary on the data from
figure [fig:statLearnFig13](fig:statLearnFig13), using $K = 10$. The
Bayes decision boundary is shown as a dashed line. The KNN and Bayes
decision boundaries are very
similar.](figures/fig2_15.png "statLearnFig15")

![A comparison of the KNN decision boundaries (solid curves) obtained
using $K=1$ and $K=100$ on the data from figure
[fig:statLearnFig13](fig:statLearnFig13). With $K=1$, the decision
boundary is overly flexible, while with $K=100$ it is not sufficiently
flexible. The Bayes decision boundary is shown as dashed
line.](figures/fig2_16.png "statLearnFig16")

In figure [fig:statLearnFig17](fig:statLearnFig17) we have plotted the
KNN test and training errors as a function of $\frac{1}{K}$. As
$\frac{1}{K}$ increases, the method becomes more flexible. As in the
regression setting, the training error rate consistently declines as the
flexibility increases. However, the test error exhibits the
characteristic U-shape, declining at first (with a minimum at
approximately $K=10$) before increasing again when the method becomes
excessively flexible and overfits.

![The KNN training error rate (blue, 200 observations) and test error
rate (orange, 5,000 observations) on the data from figure
[fig:statLearnFig13](fig:statLearnFig13) as the level of flexibility
(assessed using $\frac{1}{K}$) increases, or equivalently as the number
of neighbors $K$ decreases. The black dashed line indicates the Bayes
error rate.](figures/fig2_17.png "statLearnFig17")

\FloatBarrier

Lab: Introduction to Python
---------------------------

### Basic Commands

In `Python` a list can be created by enclosing comma-separated elements
by square brackets. Length of a list can be obtained using `len`
function.

``` {.python exports="both" results="output"}
x = [1, 3, 2, 5]
print(len(x))
y = 3
z = 5
print(y + z)
```

``` {.example}
4
8
```

To create an array of numbers, use `array` function in `numpy` library.
`numpy` functions can be used to perform element-wise operations on
arrays.

``` {.python exports="both" results="output"}
import numpy as np
x = np.array([[1, 2], [3, 4]])
y = np.array([6, 7, 8, 9]).reshape((2, 2))
print(x)
print(y)
print(x ** 2)
print(np.sqrt(y))
```

``` {.example}
[[1 2]
 [3 4]]
[[6 7]
 [8 9]]
[[ 1  4]
 [ 9 16]]
[[2.44948974 2.64575131]
 [2.82842712 3.        ]]
```

`numpy.random` has a number of functions to generate random variables
that follow a given distribution. Here we create two correlated sets of
numbers, `x` and `y`, and use `numpy.corrcoef` to calculate correlation
between them.

``` {.python exports="both" results="output"}
import numpy as np
np.random.seed(911)
x = np.random.normal(size=50)
y = x + np.random.normal(loc=50, scale=0.1, size=50)
print(np.corrcoef(x, y))
print(np.corrcoef(x, y)[0, 1])
print(np.mean(x))
print(np.var(y))
print(np.std(y) ** 2)
```

``` {.example}
[[1.         0.99374931]
 [0.99374931 1.        ]]
0.9937493134584551
-0.020219724397254404
0.9330621750073689
0.9330621750073688
```

### Graphics

`matplotlib` library has a number of functions to plot data in `Python`.
It is possible to view graphs on screen or save them in file for
inclusion in a document.

``` {.python exports="code" results="none"}
import numpy as np
import matplotlib               # only if we need to save figure in file
matplotlib.use('Agg')           # only to save figure in file
import matplotlib.pyplot as plt

x = np.random.normal(size=100)
y = np.random.normal(size=100)
plt.plot(x, y)
plt.xlabel('This is x-axis')
plt.ylabel('This is y-axis')
plt.title('Plot of X vs Y')

plt.savefig('xyPlot.png')       # only to save figure in a file
```

`numpy` function `linspace` can be used to create a sequence between a
start and an end of a given length.

``` {.python exports="code" results="none"}
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, num=50)
y = x
xx, yy = np.meshgrid(x, y)
zz = np.cos(yy) / (1 + xx ** 2)

plt.contour(xx, yy, zz)

fig, ax = plt.subplots()
zza = (zz - zz.T) / 2.0
CS = ax.contour(xx, yy, zza)
ax.clabel(CS, inline=1)
```

### Indexing Data

To access elements of an array, specify indexes inside square brackets.
It is possible to access multiple rows and columns. `shape` method gives
number of rows followed by number of columns.

``` {.python exports="both" results="output"}
import numpy as np

A = np.array(np.arange(1, 17))
A = A.reshape(4, 4, order='F')  # column first, Fortran style
print(A)
print(A[1, 2])
print(A[(0,2),:][:,(1,3)])
print(A[range(0,3),:][:,range(1,4)])
print(A[range(0, 2), :])
print(A[:, range(0, 2)])
print(A[0,:])
print(A.shape)
```

``` {.example}
[[ 1  5  9 13]
 [ 2  6 10 14]
 [ 3  7 11 15]
 [ 4  8 12 16]]
10
[ 5 15]
[ 5 10 15]
[[ 1  5  9 13]
 [ 2  6 10 14]]
[[1 5]
 [2 6]
 [3 7]
 [4 8]]
(4, 4)
```

### Loading Data

`pandas` library provides `read_csv` function to read files with data in
rectangular shape.

``` {.python exports="both" results="output"}
import pandas as pd
Auto = pd.read_csv('data/Auto.csv')
print(Auto.head())
print(Auto.shape)
print(Auto.columns)
```

``` {.example}
    mpg  cylinders  displacement  ... year  origin                       name
0  18.0          8         307.0  ...   70       1  chevrolet chevelle malibu
1  15.0          8         350.0  ...   70       1          buick skylark 320
2  18.0          8         318.0  ...   70       1         plymouth satellite
3  16.0          8         304.0  ...   70       1              amc rebel sst
4  17.0          8         302.0  ...   70       1                ford torino

[5 rows x 9 columns]
(397, 9)
Index(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'year', 'origin', 'name'],
      dtype='object')
```

To load data from an `R` library, use `get_rdataset` function from
`statsmodels`. This function seems to work only if the computer is
connected to the internet.

``` {.python exports="both" results="output"}
from statsmodels import datasets
carseats = datasets.get_rdataset('Carseats', package='ISLR').data
print(carseats.shape)
print(carseats.columns)
```

``` {.example}
(400, 11)
Index(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US'],
      dtype='object')
```

### Additional Graphical and Numerical Summaries

`plot` method can be directly applied to a `pandas` dataframe.

``` {.python exports="code" results="none"}
import pandas as pd
Auto = pd.read_csv('data/Auto.csv')
Auto.boxplot(column='mpg', by='cylinders', grid=False)
```

`hist` method can be applied to plot a histogram.

``` {.python exports="code" results="none"}
import pandas as pd
Auto = pd.read_csv('data/Auto.csv')
Auto.hist(column='mpg')
Auto.hist(column='mpg', color='red')
Auto.hist(column='mpg', color='red', bins=15)
```

For pairs plot, use `scatter_matrix` method in `pandas.plotting`.

``` {.python exports="code" results="none"}
import pandas as pd
from pandas import plotting
Auto = pd.read_csv('data/Auto.csv')
plotting.scatter_matrix(Auto[['mpg', 'displacement', 'horsepower', 'weight',
              'acceleration']])
```

On `pandas` dataframes, `describe` method produces a summary of each
variable.

``` {.python exports="both" results="output"}
import pandas as pd
Auto = pd.read_csv('data/Auto.csv')
print(Auto.describe())
```

``` {.example}
              mpg   cylinders  ...        year      origin
count  397.000000  397.000000  ...  397.000000  397.000000
mean    23.515869    5.458438  ...   75.994962    1.574307
std      7.825804    1.701577  ...    3.690005    0.802549
min      9.000000    3.000000  ...   70.000000    1.000000
25%     17.500000    4.000000  ...   73.000000    1.000000
50%     23.000000    4.000000  ...   76.000000    1.000000
75%     29.000000    8.000000  ...   79.000000    2.000000
max     46.600000    8.000000  ...   82.000000    3.000000

[8 rows x 7 columns]
```

\FloatBarrier

Linear Regression
=================

Simple Linear Regression
------------------------

Figure [fig:linearRegFig1](fig:linearRegFig1) displays the simple linear
regression fit to the `Advertising` data, where $\hat{\beta_0} =$
{{{beta0~est~}}} and $\hat{\beta_1} =$ {{{beta1~est~}}}.

![For the `Advertising` data, the least squares fit for the regression
of `sales` onto `TV` is shown. The fit is found by minimizing the sum of
squared errors. Each grey line represents an error, and the fit makes a
compromise by averaging their squares. In this case a linear fit
captures the essence of the relationship, although it is somewhat
deficient in the left of the plot.](figures/fig3_1.png "linearRegFig1")

::: {.RESULTS .drawer}
:::

In figure [fig:linearRegFig2](fig:linearRegFig2), we have computed RSS
for a number of values of $\beta_0$ and $\beta_1$, using the advertising
data with `sales` as the response and `TV` as the predictor.

![Contour and three-dimensional plots of the RSS on the `Advertising`
data, using `sales` as the response and `TV` as the predictor. The red
dots correspond to the least squares estimates $\hat{\beta_0}$ and
$\hat{\beta_1}$.](figures/fig3_2.png "linearRegFig2")

The left-hand panel of figure [fig:linearRegFig3](fig:linearRegFig3)
displays *population regression line* and *least squares line* for a
simple simulated example. The red line in the left-hand panel displays
the *true* relationship, $f(X) = 2 + 3X$, while the blue line is the
least squares estimate based on observed data. In the right-hand panel
of figure [fig:linearRegFig3](fig:linearRegFig3) we have generated five
different data sets from the model $Y = 2 + 3X + \epsilon$ and plotted
the corresponding five least squares lines.

![A simulated data set. Left: The red line represents the true
relationship, $f(X) = 2 + 3X$, which is known as the population
regression line. The blue line is the least squares line; it is the
least squares estimate for $f(X)$ based on the observed data, shown in
grey circles. Right: The population regression line is again shown in
red, and the least squares line in blue. In cyan, five least squares
lines are shown, each computed on the basis of a separate random set of
observations. Each least squares line is different, but on average, the
least squares lines are quite close to the population regression
line.](figures/fig3_3.png "linearRegFig3")

\FloatBarrier

For `Advertising` data, table [tab:linearRegTab1](tab:linearRegTab1)
provides details of the least squares model for the regression of number
of units sold on TV advertising budget.

\bigskip

              Coef.    Std.Err.   $t$       $P > \mid t \mid$
  ----------- -------- ---------- --------- -------------------
  Intercept   7.0326   0.4578     15.3603   0.0
  TV          0.0475   0.0027     17.6676   0.0

  : For `Advertising` data, the coefficients of the least squares model
  for the regression of number of units sold on TV advertising budget.
  An increase of \$1,000 on the TV advertising budget is associated with
  an increase in sales by around 50 units.

\bigskip

Next, in table [tab:linearRegTab2](tab:linearRegTab2), we report more
information about the least squares model.

  Quantity                  Value
  ------------------------- ---------
  Residual standard error   3.259
  $R^2$                     0.612
  F-statistic               312.145

  : For the `Advertising` data, more information about the least squares
  model for the regression of number of units sold on TV advertising
  budget.

\FloatBarrier

Multiple Linear Regression
--------------------------

Table [tab:linearRegTab3](tab:linearRegTab3) and the next table show
results of two simple linear regressions, each of which uses a different
advertising medium as a predictor. We find that a \$1,000 increase in
spending on radio advertising is associated with an increase in sales by
around {{{radio~betaest~}}} units. A \$1,000 increase in advertising
spending on on newspapers increases sales by approximately
{{{newsp~betaest~}}} units.

              Coef.   Std.Err.   $t$      $P > \mid t \mid$
  ----------- ------- ---------- -------- -------------------
  Intercept   9.312   0.563      16.542   0.0
  radio       0.202   0.02       9.921    0.0

  : More simple linear regression models for `Advertising` data.
  Coefficients of the simple linear regression model for number of units
  sold on radio advertising budget. a \$1,000 increase in spending on
  radio advertising is associated with an average increase sales by
  around {{{radio~betaest~}}} units.

              Coef.    Std.Err.   $t$      $P > \mid t \mid$
  ----------- -------- ---------- -------- -------------------
  Intercept   12.351   0.621      19.876   0.0
  newspaper   0.055    0.017      3.3      0.001

\FloatBarrier

::: {.RESULTS .drawer}
:::

Figure [fig:linearRegFig4](fig:linearRegFig4) illustrates an example of
the least squares fit to a toy data set with $p = 2$ predictors.

![In a three-dimensional setting, with two predictors and one response,
the least squares regression line becomes a plane. The plane is chosen
to minimize the sum of the squared vertical distances between each
observation (shown in red) and the
plane.](figures/fig3_4.png "linearRegFig4")

Table [tab:linearRegTab4](tab:linearRegTab4) displays multiple
regression coefficient estimates when TV, radio, and newspaper
advertising budgets are used to predict product sales using
`Advertising` data.

              Coef.    Std.Err.   $t$      $P > \mid t \mid$
  ----------- -------- ---------- -------- -------------------
  Intercept   2.939    0.312      9.422    0.0
  TV          0.046    0.001      32.809   0.0
  radio       0.189    0.009      21.893   0.0
  newspaper   -0.001   0.006      -0.177   0.86

  : For the `Advertising` data, least squares coefficient estimates of
  the multiple linear regression of number of units sold on radio, TV,
  and newspaper advertising budgets.

Table [tab:linearRegTab5](tab:linearRegTab5) shows the correlation
matrix for the three predictor variables and response variable in table
[tab:linearRegTab4](tab:linearRegTab4).

              TV       radio    newspaper   sales
  ----------- -------- -------- ----------- --------
  TV          1.0      0.0548   0.0566      0.7822
  radio       0.0548   1.0      0.3541      0.5762
  newspaper   0.0566   0.3541   1.0         0.2283
  sales       0.7822   0.5762   0.2283      1.0

  : Correlation matrix for `TV`, `radio`, and `sales` for the
  `Advertising` data.

  Quantity                  Value
  ------------------------- -------
  Residual standard error   1.69
  $R^2$                     0.897
  F-statistic               570.0

  : More information about the least squares model for the regression of
  number of units sold on TV, newspaper, and radio advertising budgets
  in the `Advertising` data. Other information about this model was
  displayed in table [tab:linearRegTab4](tab:linearRegTab4).

Figure [fig:linearRegFig5](fig:linearRegFig5) displays a
three-dimensional plot of `TV` and `radio` versus `sales`.

![For the `Advertising` data, a linear regression fit to `sales` using
`TV` and `radio` as predictors. From the pattern of the residuals, we
can see that there is a pronounced non-linear relationship in the data.
The positive residuals tend to lie along the 45-degree line, where TV
and Radio budgets are split evenly. The negative residuals tend to lie
away from this line, where budgets are more
lopsided.](figures/fig3_5.png "linearRegFig5")

\FloatBarrier

Other Considerations in the Regression Model
--------------------------------------------

`Credit` data set displayed in figure
[fig:linearRegFig3\_6](fig:linearRegFig3_6) records `balance` (average
credit card debt for a number of individuals) as well as several
quantitative predictors: `age`, `cards` (number of credit cards),
`education` and `rating` (credit rating).

![The `Credit` dataset contains information about `balance`, `age`,
`cards`, `education`, `income`, `limit`, and `rating` for a number of
potential customers.](figures/fig3_6.png "linearRegFig3_6")

Table [tab:linearRegTab7](tab:linearRegTab7) displays the coefficient
estimates and other information associated with the model where `gender`
is the only explanatory variable.

                       Coef.     Std.Err.   $t$      $P > \mid t \mid$
  -------------------- --------- ---------- -------- -------------------
  Intercept            509.803   33.128     15.389   0.0
  Gender\[T.Female\]   19.733    46.051     0.429    0.669

  : Least squares coefficient estimates associated with the regression
  of `balance` onto `gender` in the `Credit` data set.

From table [tab:linearRegTab8](tab:linearRegTab8) we see that the
estimated `balance` for the baseline, African American, is
\${{{afr~amrest~}}}. It is estimated that the Asian category will have
an additional \${{{asian~incr~}}} debt, and that the Caucasian category
will have an additional \${{{cauc~incr~}}} debt compared to Africna
American category.

                             Coef.     Std.Err.   $t$      $P > \mid t \mid$
  -------------------------- --------- ---------- -------- -------------------
  Intercept                  531.0     46.319     11.464   0.0
  Ethnicity\[T.Asian\]       -18.686   65.021     -0.287   0.774
  Ethnicity\[T.Caucasian\]   -12.503   56.681     -0.221   0.826

  : Least squares coefficient estimates associated with the regression
  of `balance` onto `ethnicity` in the `Credit` data set.

::: {.RESULTS .drawer}
:::

Table [tab:linearRegTab9](tab:linearRegTab9) shows results of regressing
`sales` and `TV` and `radio` when an interaction term is included.
Coefficient of interaction term `TV:radio` is highly significant.

In figure [fig:linearRegFig7](fig:linearRegFig7), the left panel shows
least squares lines when we predict `balance` using `income`
(quantitative) and `student` (qualitative variables). There is no
interaction term between `income` and `student`. The right panel shows
least squares lines when an interaction term is included.

               Coef.   Std.Err.   $t$      $P > \mid t \mid$
  ------------ ------- ---------- -------- -------------------
  Intercept    6.75    0.248      27.233   0.0
  TV           0.019   0.002      12.699   0.0
  radio        0.029   0.009      3.241    0.001
  <TV:radio>   0.001   0.0        20.727   0.0

  : For `Advertising` data, least squares coefficient estimates
  associated with the regression of `sales` onto `TV` and `radio`, with
  an interaction term.

![For the `Credit` data, the least squares lines are shown for
prediction of `balance` from `income` for students and non-students.
Left: There is no interaction between `income` and `student`. Right:
There is an interaction term between `income` and
`students`.](figures/fig3_7.png "linearRegFig7")

Figure [fig:linearRegFig8](fig:linearRegFig8) shows a scatter plot of
`mpg` (gas mileage in miles per gallon) versus `horsepower` in the
`Auto` data set. The figure also includes least squares fit line for
linear, second degree, and fifth degree polynomials in `horsepower`.

![The `Auto` data set. For a number of cars, `mpg` and `horsepower` are
shown. The linear regression fit is shown in orange. The linear
regression fit for a model that includes first- and second-order terms
of `horsepower` is shown as blue curve. The linear regression fit for a
model that includes all polynomials of `horsepower` up to fifth-degree
is shown in green.](figures/fig3_8.png "linearRegFig8")

\FloatBarrier

Lab: Linear Regression
----------------------

It is possible to call R from Python and *vice versa*.
