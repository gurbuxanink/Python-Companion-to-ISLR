
# Table of Contents

1.  [Introduction](#org8c1ffd2)
2.  [Statistical Learning](#org5f81d39)
    1.  [What is Statistical Learning?](#org18ce470)
    2.  [Assessing Model Accuracy](#orga257e5e)
    3.  [Lab: Introduction to Python](#org1289e3f)
        1.  [Basic Commands](#org501e183)
        2.  [Graphics](#orgdb9f972)
        3.  [Indexing Data](#orgf91e1fc)
        4.  [Loading Data](#orgaae22e9)
        5.  [Additional Graphical and Numerical Summaries](#org0b437a9)
3.  [Linear Regression](#orgfdf68cd)
    1.  [Simple Linear Regression](#org8ff803b)
    2.  [Multiple Linear Regression](#orge49d1d6)
    3.  [Other Considerations in the Regression Model](#org8a16dd3)
    4.  [The Marketing Plan](#org795e061)
    5.  [Comparison of Linear Regression with K-Nearest Neighbors](#org133cfd7)
    6.  [Lab: Linear Regression](#org2af7f1e)
        1.  [Libraries](#org5ea7a3c)
        2.  [Simple Linear Regression](#orgf31ef28)
        3.  [Multiple Linear Regression](#org4d8f496)
        4.  [Interaction Terms](#orgae8db59)
        5.  [Non-linear Transformations of the Predictors](#org166f20d)
        6.  [Qualitative Predictors](#orga3ed622)
        7.  [Calling `R` from `Python`](#orgcddf23b)
4.  [Classification](#org0943dd8)
    1.  [An Overview of Classification](#orgb649a04)
    2.  [Why Not Linear Regression?](#org8afbfa7)
    3.  [Logistic Regression](#orgbbdfc0d)
    4.  [Linear Discriminant Analysis](#org6a9ceec)
    5.  [A Comparison of Classification Methods](#orgb5841fc)
    6.  [Lab: Logistic Regression, LDA, QDA, and KNN](#orga0e83c5)
        1.  [The Stock Market Data](#org34b2dd2)
        2.  [Logistc Regression](#orgf525a1b)
        3.  [Linear Discriminant Analysis](#org4f2ae5b)
        4.  [Quadratic Discriminant Analysis](#org79fb297)
        5.  [K-Nearest Neightbors](#org657807b)
        6.  [An Application to Caravan Insurance Data](#org104c999)
5.  [Resampling Methods](#orga6304fb)
    1.  [Cross-Validation](#org0c4d8a0)
    2.  [The Bootstrap](#org8d2c9b6)



<a id="org8c1ffd2"></a>

# Introduction

Figure [2](#org7d7c8b9) shows graphs of Wage versus three variables. 

![img](figures/fig1_1.png "`Wage` data, which contains income survey information for males from the central Atlantic region of the United States.  Left: `wage` as a function of `age`.  On average, `wage` increases with `age` until about 60 years of age, at which point it begins to decline.  Center: `wage` as a function of `year`.  There is a slow but steady increase of approximately $10,000 in the average `wage` between 2003 and 2009.  Right: Boxplots displaying `wage` as a function of `education`, with 1 indicating the lowest level (no highschool diploma) and 5 the highest level (an advanced graduate degree).  On average, `wage` increases with the level of `education`.")

Figure [4](#org868b9bd) shows boxplots of previous days' percentage changes in S&P
500 grouped according to today's change `Up` or `Down`. 

![img](figures/fig1_2.png "Left: Boxplots of the previous day's percentage change in the S&P 500 index for the days for which the market increased or decreased, obtained from the `Smarket` data.  Center and Right: Same as left panel, but the percentage changes for two and three days previous are shown.")


<a id="org5f81d39"></a>

# Statistical Learning

<a id="org4192780"></a>


<a id="org18ce470"></a>

## What is Statistical Learning?

Figure [7](#orgc80f281) shows scatter plots of `sales` versus `TV`, `radio`,
and `newspaper` advertising.  In each panel, the figure also includes an OLS
regression line.  

![img](figures/fig2_1.png "The `Advertising` data set. The plot displays `sales`, in thousands of units, as a function of `TV`, `radio`, and `newspaper` budgets, in thousands of dollars, for 200 different markets.  In each plot we show the simple least squares fit of `sales` to that variable.  In other words, each red line represents a simple model that can be used to predict `sales` using `TV`, `radio`, and `newspaper`, respectively.")

Figure [9](#org8f1d44b) is a plot of `Income` versus `Years of Education` from the
Income data set.  In the left panel, the \`\`true'' function (given by blue line)
is actually my guess.  

![img](figures/fig2_2.png "The `Income` data set.  Left: The red dots are the observed values of `income` (in tens of thousands of dollars) and `years of education` for 30 individuals.  Right: The blue curve represents the true underlying relationship between `income` and `years of education`, which is generally unknown (but is known in this case because the data are simulated).  The vertical lines represent the error associated with each observation.  Note that some of the errors are positive (when an observation lies above the blue curve) and some are negative (when an observation lies below the curve).  Overall, these errors have approximately mean zero.")

Figure [11](#org950bdcb) is a plot of `Income` versus `Years of Education` and
`Seniority` from the `Income` data set.  Since the book does not provide the
true values of `Income`, \`\`true'' values shown in the plot are actually third
order polynomial fit.  

![img](figures/fig2_3.png "The plot displays `income` as a function of `years of education` and `seniority` in the `Income` data set.  The blue surface represents the true underlying relationship between `income` and `years of education` and `seniority`, which is known since the data are simulated.  The red dots indicate the observed values of these quantities for 30 individuals.")

Figure [13](#orgd4da255) shows an example of the parametric approach applied to
the `Income` data from previous figure. 

![img](figures/fig2_4.png "A linear model fit by least squares to the `Income` data from figure [11](#org950bdcb).  The observations are shown in red, and the blue plane indicates the least squares fit to the data.")

Figure [15](#org597b8c0) provides an illustration of the trade-off between
flexibility and interpretability for some of the methods covered in this book.

![img](figures/figure2_7.png "A representation of the tradeoff between flexibility and interpretability, using different statistical learning methods.  In general, as the flexibility of a method increases, its interpretability decreases.")

Figure [17](#orgf028e69) provides a simple illustration of the clustering problem.

![img](figures/fig2_8.png "A clustering data set involving three groups.  Each group is shown using a different colored symbol.  Left: The three groups are well-separated.  In this setting, a clustering approach should successfully identify the three groups.  Right: There is some overlap among the groups.  Now the clustering taks is more challenging.")


<a id="orga257e5e"></a>

## Assessing Model Accuracy

Figure [21](#orgf001635) illustrates the tradeoff between training MSE and test
MSE.  We select a \`\`true function'' whose shape is similar to that shown in the
book.  In the left panel, the orange, blue, and green curves illustrate three possible estimates
for \(f\) given by the black curve.  The orange line is the linear regression
fit, which is relatively inflexible.  The blue and green curves were produced
using *smoothing splines* from `UnivariateSpline` function in `scipy` package.
We obtain different levels of flexibility by varying the parameter `s`, which
affects the number of knots.  

For the right panel, we have chosen polynomial fits.  The degree of polynomial
represents the level of flexibility.  This is because the function
`UnivariateSpline` does not more than five degrees of freedom.  

When we repeat the simulations for figure [21](#orgf001635), we see considerable
variation in the right panel MSE plots.  But the overall conclusion remains the
same.   

![img](figures/fig2_9.png "Left: Data simulated from \(f\), shown in black.  Three estimates of \(f\) are shown: the linear regression line (orange curve), and two smoothing spline fits (blue and green curves).  Right: Training MSE (grey curve), test MSE (red curve), and minimum possible test MSE over all methods (dashed grey line).")

Figure [23](#orga1b7587) provides another example in which the true \(f\) is
approximately linear. 

![img](figures/fig2_10.png "Details are as in figure [21](#orgf001635) using a different true \(f\) that is much closer to linear.  In this setting, linear regression provides a very good fit to the data.")

Figure [25](#org662b33c) displays an example in which \(f\) is highly
non-linear. The training and test MSE curves still exhibit the same general
patterns.

![img](figures/fig2_11.png "Details are as in figure [21](#orgf001635), using a different \(f\) that is far from linear.  In this setting, linear regression provides a very poor fit to the data.")

Figure [27](#org47ffb4a) displays the relationship between bias, variance, and
test MSE.  This relationship is referred to as *bias-variance trade-off*.  When
simulations are repeated, we see considerable variation in different graphs,
especially for MSE lines.  But overall shape remains the same. 

![img](figures/fig2_12.png "Squared bias (blue curve), variance (orange curve), \(Var(\epsilon)\) (dashed line), and test MSE (red curve) for the three data sets in figures [21](#orgf001635) - [25](#org662b33c).  The vertical dotted line indicates the flexibility level corresponding to the smallest test MSE.")

Figure [29](#org76384cb) provides an example using a simulated data set in
two-dimensional space consisting of predictors \(X_1\) and \(X_2\).  

![img](figures/fig2_13.png "A simulated data set consisting of 200 observations in two groups, indicated in blue and orange.  The dashed line represents the Bayes decision boundary.  The orange background grid indicates the region in which a test observation will be assigned to the orange class, and blue background grid indicates the region in which a test observation will be assigned to the blue class.")

Figure [31](#orgd19e6dd) displays the KNN decision boundary, using \(K=10\), when
applied to the simulated data set from figure [29](#org76384cb).  Even though
the true distribution is not known by the KNN classifier, the KNN decision
making boundary is very close to that of the Bayes classifier.  

![img](figures/fig2_15.png "The firm line indicates the KNN decision boundary on the data from figure [29](#org76384cb), using \(K = 10\). The Bayes decision boundary is shown as a dashed line.  The KNN and Bayes decision boundaries are very similar.")

![img](figures/fig2_16.png "A comparison of the KNN decision boundaries (solid curves) obtained using \(K=1\) and \(K=100\) on the data from figure [29](#org76384cb).  With \(K=1\), the decision boundary is overly flexible, while with \(K=100\) it is not sufficiently flexible.  The Bayes decision boundary is shown as dashed line.")

In figure [34](#org7bd8572) we have plotted the KNN test and training errors as
a function of \(\frac{1}{K}\).  As \(\frac{1}{K}\) increases, the method becomes
more flexible.  As in the regression setting, the training error rate
consistently declines as the flexibility increases.  However, the test error
exhibits the characteristic U-shape, declining at first (with a minimum at
approximately \(K=10\)) before increasing again when the method becomes
excessively flexible and overfits. 

![img](figures/fig2_17.png "The KNN training error rate (blue, 200 observations) and test error rate (orange, 5,000 observations) on the data from figure [29](#org76384cb) as the level of flexibility (assessed using \(\frac{1}{K}\)) increases, or equivalently as the number of neighbors \(K\) decreases.  The black dashed line indicates the Bayes error rate.")


<a id="org1289e3f"></a>

## Lab: Introduction to Python


<a id="org501e183"></a>

### Basic Commands

In `Python` a list can be created by enclosing comma-separated elements by
square brackets.  Length of a list can be obtained using `len` function.

    x = [1, 3, 2, 5]
    print(len(x))
    y = 3
    z = 5
    print(y + z)

    4
    8

To create an array of numbers, use `array` function in `numpy` library.  `numpy`
functions can be used to perform element-wise operations on arrays.

    import numpy as np
    x = np.array([[1, 2], [3, 4]])
    y = np.array([6, 7, 8, 9]).reshape((2, 2))
    print(x)
    print(y)
    print(x ** 2)
    print(np.sqrt(y))

    [[1 2]
     [3 4]]
    [[6 7]
     [8 9]]
    [[ 1  4]
     [ 9 16]]
    [[2.44948974 2.64575131]
     [2.82842712 3.        ]]

`numpy.random` has a number of functions to generate random variables that
follow a given distribution.  Here we create two correlated sets of numbers, `x`
and `y`, and use `numpy.corrcoef` to calculate correlation between them. 

    import numpy as np
    np.random.seed(911)
    x = np.random.normal(size=50)
    y = x + np.random.normal(loc=50, scale=0.1, size=50)
    print(np.corrcoef(x, y))
    print(np.corrcoef(x, y)[0, 1])
    print(np.mean(x))
    print(np.var(y))
    print(np.std(y) ** 2)

    [[1.         0.99374931]
     [0.99374931 1.        ]]
    0.9937493134584551
    -0.020219724397254404
    0.9330621750073689
    0.9330621750073688


<a id="orgdb9f972"></a>

### Graphics

`matplotlib` library has a number of functions to plot data in `Python`.  It is
possible to view graphs on screen or save them in file for inclusion in a
document. 

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

`numpy` function `linspace` can be used to create a sequence between a start and
an end of a given length.  

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


<a id="orgf91e1fc"></a>

### Indexing Data

To access elements of an array, specify indexes inside square brackets.  It is
possible to access multiple rows and columns. `shape` method gives number of
rows followed by number of columns. 

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


<a id="orgaae22e9"></a>

### Loading Data

`pandas` library provides `read_csv` function to read files with data in
rectangular shape.  

    import pandas as pd
    Auto = pd.read_csv('data/Auto.csv')
    print(Auto.head())
    print(Auto.shape)
    print(Auto.columns)

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

To load data from an `R` library, use `get_rdataset` function from
`statsmodels`.  This function seems to work only if the computer is connected to
the internet. 

    from statsmodels import datasets
    carseats = datasets.get_rdataset('Carseats', package='ISLR').data
    print(carseats.shape)
    print(carseats.columns)

    (400, 11)
    Index(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
           'ShelveLoc', 'Age', 'Education', 'Urban', 'US'],
          dtype='object')


<a id="org0b437a9"></a>

### Additional Graphical and Numerical Summaries

`plot` method can be directly applied to a `pandas` dataframe.  

    import pandas as pd
    Auto = pd.read_csv('data/Auto.csv')
    Auto.boxplot(column='mpg', by='cylinders', grid=False)

`hist` method can be applied to plot a histogram. 

    import pandas as pd
    Auto = pd.read_csv('data/Auto.csv')
    Auto.hist(column='mpg')
    Auto.hist(column='mpg', color='red')
    Auto.hist(column='mpg', color='red', bins=15)

For pairs plot, use `scatter_matrix` method in `pandas.plotting`.  

    import pandas as pd
    from pandas import plotting
    Auto = pd.read_csv('data/Auto.csv')
    plotting.scatter_matrix(Auto[['mpg', 'displacement', 'horsepower', 'weight',
    			      'acceleration']])

On `pandas` dataframes, `describe` method produces a summary of each variable. 

    import pandas as pd
    Auto = pd.read_csv('data/Auto.csv')
    print(Auto.describe())

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


<a id="orgfdf68cd"></a>

# Linear Regression


<a id="org8ff803b"></a>

## Simple Linear Regression

Figure [48](#org87b4d76) displays the simple linear regression fit to the
`Advertising` data, where \(\hat{\beta_0} =\) 0.0475
 and \(\hat{\beta_1} =\) 7.0326.

![img](figures/fig3_1.png "For the `Advertising` data, the least squares fit for the regression of `sales` onto `TV` is shown.  The fit is found by minimizing the sum of squared errors.  Each grey line represents an error, and the fit makes a compromise by averaging their squares.  In this case a linear fit captures the essence of the relationship, although it is somewhat deficient in the left of the plot.")


In figure [50](#orgccb4a78), we have computed RSS for a number of values of
\(\beta_0\) and \(\beta_1\), using the advertising data with `sales` as the response
and `TV` as the predictor. 

![img](figures/fig3_2.png "Contour and three-dimensional plots of the RSS on the `Advertising` data, using `sales` as the response and `TV` as the predictor.  The red dots correspond to the least squares estimates \(\hat{\beta_0}\) and \(\hat{\beta_1}\).")

The left-hand panel of figure [52](#org7d7aa16) displays *population regression
line* and *least squares line* for a simple simulated example.  The red line in
the left-hand panel displays the *true* relationship, \(f(X) = 2 + 3X\), while the
blue line is the least squares estimate based on observed data.  In the
right-hand panel of figure [52](#org7d7aa16) we have generated five different
data sets from the model \(Y = 2 + 3X + \epsilon\) and plotted the corresponding
five least squares lines.  

![img](figures/fig3_3.png "A simulated data set.  Left: The red line represents the true relationship, \(f(X) = 2 + 3X\), which is known as the population regression line.  The blue line is the least squares line; it is the least squares estimate for \(f(X)\) based on the observed data, shown in grey circles.  Right: The population regression line is again shown in red, and the least squares line in blue.  In cyan, five least squares lines are shown, each computed on the basis of a separate random set of observations.  Each least squares line is different, but on average, the least squares lines are quite close to the population regression line.")

For `Advertising` data, table [1](#orgf30100f) provides details of the least squares model for the
regression of number of units sold on TV advertising budget. 

<table id="orgf30100f" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 1:</span> For `Advertising` data, the coefficients of the least squares model for the regression of number of units sold on TV advertising budget.  An increase of $1,000 on the TV advertising budget is associated with an increase in sales by around 50 units.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(t\)</th>
<th scope="col" class="org-right">\(P > \mid t \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">7.0326</td>
<td class="org-right">0.4578</td>
<td class="org-right">15.3603</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">TV</td>
<td class="org-right">0.0475</td>
<td class="org-right">0.0027</td>
<td class="org-right">17.6676</td>
<td class="org-right">0.0</td>
</tr>
</tbody>
</table>

Next, in table [2](#org8a3e33b), we report more information about the least squares model.  

<table id="org8a3e33b" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 2:</span> For the `Advertising` data, more information about the least squares model for the regression of number of units sold on TV advertising budget.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Quantity</th>
<th scope="col" class="org-right">Value</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Residual standard error</td>
<td class="org-right">3.259</td>
</tr>


<tr>
<td class="org-left">\(R^2\)</td>
<td class="org-right">0.612</td>
</tr>


<tr>
<td class="org-left">F-statistic</td>
<td class="org-right">312.145</td>
</tr>
</tbody>
</table>


<a id="orge49d1d6"></a>

## Multiple Linear Regression

Table [3](#org33993e4)  shows results of two simple linear
regressions, each of which uses a different advertising medium as a predictor.
We find that a $1,000 increase in spending on radio advertising is associated
with an increase in sales by around 202 units.  A $1,000 increase in advertising
spending on on newspapers increases sales by approximately 55 units. 

<table id="org33993e4" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 3:</span> More simple linear regression models for `Advertising` data.  Coefficients of the simple linear regression model for number of units sold on Top: radio advertising budget and Bottom: newspaper advertising budget. A $1,000 increase in spending on radio advertising is associated with an average increase sales by around 202 units, while the same increase in spending on newspaper advertising is associated with an average increase of around 55 units.  `Sales` variable is in thousands of units, and the `radio` and `newspaper` variables are in thousands of dollars..</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(t\)</th>
<th scope="col" class="org-right">\(P > \mid t \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">9.312</td>
<td class="org-right">0.563</td>
<td class="org-right">16.542</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">radio</td>
<td class="org-right">0.202</td>
<td class="org-right">0.02</td>
<td class="org-right">9.921</td>
<td class="org-right">0.0</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">12.351</td>
<td class="org-right">0.621</td>
<td class="org-right">19.876</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">newspaper</td>
<td class="org-right">0.055</td>
<td class="org-right">0.017</td>
<td class="org-right">3.3</td>
<td class="org-right">0.001</td>
</tr>
</tbody>
</table>


Figure [57](#orga3dc5a3) illustrates an example of the least squares fit to a
toy data set with \(p = 2\) predictors. 

![img](figures/fig3_4.png "In a three-dimensional setting, with two predictors and one response, the least squares regression line becomes a plane.  The plane is chosen to minimize the sum of the squared vertical distances between each observation (shown in red) and the plane.")

Table [4](#orgda8acc6) displays multiple regression coefficient estimates when
TV, radio, and newspaper advertising budgets are used to predict product sales
using `Advertising` data.

<table id="orgda8acc6" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 4:</span> For the `Advertising` data, least squares coefficient estimates of the multiple linear regression of number of units sold on radio, TV, and newspaper advertising budgets.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(t\)</th>
<th scope="col" class="org-right">\(P > \mid t \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">2.939</td>
<td class="org-right">0.312</td>
<td class="org-right">9.422</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">TV</td>
<td class="org-right">0.046</td>
<td class="org-right">0.001</td>
<td class="org-right">32.809</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">radio</td>
<td class="org-right">0.189</td>
<td class="org-right">0.009</td>
<td class="org-right">21.893</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">newspaper</td>
<td class="org-right">-0.001</td>
<td class="org-right">0.006</td>
<td class="org-right">-0.177</td>
<td class="org-right">0.86</td>
</tr>
</tbody>
</table>

Table [5](#org0f3977a) shows the correlation matrix for the three predictor
variables and response variable in table [4](#orgda8acc6). 

<table id="org0f3977a" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 5:</span> Correlation matrix for `TV`, `radio`, and `sales` for the `Advertising` data.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">TV</th>
<th scope="col" class="org-right">radio</th>
<th scope="col" class="org-right">newspaper</th>
<th scope="col" class="org-right">sales</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">TV</td>
<td class="org-right">1.0</td>
<td class="org-right">0.0548</td>
<td class="org-right">0.0566</td>
<td class="org-right">0.7822</td>
</tr>


<tr>
<td class="org-left">radio</td>
<td class="org-right">0.0548</td>
<td class="org-right">1.0</td>
<td class="org-right">0.3541</td>
<td class="org-right">0.5762</td>
</tr>


<tr>
<td class="org-left">newspaper</td>
<td class="org-right">0.0566</td>
<td class="org-right">0.3541</td>
<td class="org-right">1.0</td>
<td class="org-right">0.2283</td>
</tr>


<tr>
<td class="org-left">sales</td>
<td class="org-right">0.7822</td>
<td class="org-right">0.5762</td>
<td class="org-right">0.2283</td>
<td class="org-right">1.0</td>
</tr>
</tbody>
</table>

<table id="org8e7d860" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 6:</span> More information about the least squares model for the regression of number of units sold on TV, newspaper, and radio advertising budgets in the `Advertising` data.  Other information about this model was displayed in table [4](#orgda8acc6).</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Quantity</th>
<th scope="col" class="org-right">Value</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Residual standard error</td>
<td class="org-right">1.69</td>
</tr>


<tr>
<td class="org-left">\(R^2\)</td>
<td class="org-right">0.897</td>
</tr>


<tr>
<td class="org-left">F-statistic</td>
<td class="org-right">570.0</td>
</tr>
</tbody>
</table>

Figure [61](#org76d7e72) displays a three-dimensional plot of `TV` and `radio`
versus `sales`.  

![img](figures/fig3_5.png "For the `Advertising` data, a linear regression fit to `sales` using `TV` and `radio` as predictors.  From the pattern of the residuals, we can see that there is a pronounced non-linear relationship in the data.  The positive residuals tend to lie along the 45-degree line, where TV and Radio budgets are split evenly.  The negative residuals tend to lie away from this line, where budgets are more lopsided.")


<a id="org8a16dd3"></a>

## Other Considerations in the Regression Model

`Credit` data set displayed in figure [63](#org19ea572) records `balance`
(average credit card debt for a number of individuals) as well as several
quantitative predictors: `age`, `cards` (number of credit cards), `education`
and `rating` (credit rating).

![img](figures/fig3_6.png "The `Credit` dataset contains information about `balance`, `age`, `cards`, `education`, `income`, `limit`, and `rating` for a number of potential customers.")

Table [7](#org1873f6a) displays the coefficient estimates and other information
associated with the model where `gender` is the only explanatory variable.

<table id="org1873f6a" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 7:</span> Least squares coefficient estimates associated with the regression of `balance` onto `gender` in the `Credit` data set.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(t\)</th>
<th scope="col" class="org-right">\(P > \mid t \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">509.803</td>
<td class="org-right">33.128</td>
<td class="org-right">15.389</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">Gender[T.Female]</td>
<td class="org-right">19.733</td>
<td class="org-right">46.051</td>
<td class="org-right">0.429</td>
<td class="org-right">0.669</td>
</tr>
</tbody>
</table>

From table [8](#org0ffd7e8) we see that the estimated `balance` for the
baseline, African American, is $531.0. It is estimated that the
Asian category will have an additional $-18.7 debt, and that the
Caucasian category will have an additional $-12.5 debt compared to
African American category.

<table id="org0ffd7e8" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 8:</span> Least squares coefficient estimates associated with the regression of `balance` onto `ethnicity` in the `Credit` data set.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(t\)</th>
<th scope="col" class="org-right">\(P > \mid t \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">531.0</td>
<td class="org-right">46.319</td>
<td class="org-right">11.464</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">Ethnicity[T.Asian]</td>
<td class="org-right">-18.686</td>
<td class="org-right">65.021</td>
<td class="org-right">-0.287</td>
<td class="org-right">0.774</td>
</tr>


<tr>
<td class="org-left">Ethnicity[T.Caucasian]</td>
<td class="org-right">-12.503</td>
<td class="org-right">56.681</td>
<td class="org-right">-0.221</td>
<td class="org-right">0.826</td>
</tr>
</tbody>
</table>


Table [9](#orgeae7333) shows results of regressing `sales` and `TV` and `radio`
when an interaction term is included.  Coefficient of interaction term
`TV:radio` is highly significant.

In figure [68](#org9a35a81), the left panel shows least squares lines when
we predict `balance` using `income` (quantitative) and `student` (qualitative
variables). There is no interaction term between `income` and `student`.  The
right panel shows least squares lines when an interaction term is included. 

<table id="orgeae7333" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 9:</span> For `Advertising` data, least squares coefficient estimates associated with the regression of `sales` onto `TV` and `radio`, with an interaction term.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(t\)</th>
<th scope="col" class="org-right">\(P > \mid t \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">6.75</td>
<td class="org-right">0.248</td>
<td class="org-right">27.233</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">TV</td>
<td class="org-right">0.019</td>
<td class="org-right">0.002</td>
<td class="org-right">12.699</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">radio</td>
<td class="org-right">0.029</td>
<td class="org-right">0.009</td>
<td class="org-right">3.241</td>
<td class="org-right">0.001</td>
</tr>


<tr>
<td class="org-left">TV:radio</td>
<td class="org-right">0.001</td>
<td class="org-right">0.0</td>
<td class="org-right">20.727</td>
<td class="org-right">0.0</td>
</tr>
</tbody>
</table>

![img](figures/fig3_7.png "For the `Credit` data, the least squares lines are shown for prediction of `balance` from `income` for students and non-students.  Left: There is no interaction between `income` and `student`.  Right: There is an interaction term between `income` and `students`.")

Figure [70](#orgc1bae16) shows a scatter plot of `mpg` (gas mileage in miles per
gallon) versus `horsepower` in the `Auto` data set.  The figure also includes
least squares fit line for linear, second degree, and fifth degree polynomials
in `horsepower`. 

![img](figures/fig3_8.png "The `Auto` data set.  For a number of cars, `mpg` and `horsepower` are shown.  The linear regression fit is shown in orange.  The linear regression fit for a model that includes first- and second-order terms of `horsepower` is shown as blue curve.  The linear regression fit for a model that includes all polynomials of `horsepower` up to fifth-degree is shown in green.")

Table [10](#org57f2adf) shows regression results of a quadratic fit to explain
`mpg` as a function of `horsepower` and \(\mathttt{horsepower^2}\).  

<table id="org57f2adf" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 10:</span> For the `Auto` data set, least squares coefficient estimates associated with the regression of `mpg` onto `horsepower` and \(\texttt{horsepower^2}\).</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(t\)</th>
<th scope="col" class="org-right">\(P > \mid t \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">56.9001</td>
<td class="org-right">1.8004</td>
<td class="org-right">31.6037</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">horsepower</td>
<td class="org-right">-0.4662</td>
<td class="org-right">0.0311</td>
<td class="org-right">-14.9782</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">\(horsepower^2\)</td>
<td class="org-right">0.0012</td>
<td class="org-right">0.0001</td>
<td class="org-right">10.0801</td>
<td class="org-right">0.0</td>
</tr>
</tbody>
</table>

The left panel of figure [73](#org8a3fdf2) displays a residual plot from the
linear regression of `mpg` onto `horsepower` on the `Auto` data set.  The red
line is a smooth fit to the residuals, which is displayed in order to make it
easier to identify any trends.  The residuals exhibit a clear U-shape, which
strongly suggests non-linearity in the data.  In contrast, the right hand panel
of figure[73](#org8a3fdf2) displays the residual plot results from the model
which contains a quadratic term in `horsepower`.  Now there is little pattern in
residuals, suggesting that the quadratic term improves the fit to the data.

![img](figures/fig3_9.png "Plots of residuals versus predicted (or fitted) values for the `Auto` data set.  In each plot, the red line is a smooth fit to the residuals, intended to make it easier to identify a trend.  Left: A linear regression of `mpg` on `horsepower`.  A strong pattern in the residuals indicates non-linearity in the data.  Right: A linear regression of `mpg` on `horsepower` and square of `horsepower`.  Now there is little pattern in the residuals.")

Figure [75](#orge457ee0) provides an illustration of correlations among
residuals.  In the top panel, we see the residuals from a linear regression fit
to data generated with uncorrelated errors.  There is no evidence of
time-related trend in the residuals.  In contrast, the residuals in the bottom
panel are from a data set in which adjacent errors had a correlation of 0.9.
Now there is a clear pattern in the residuals - adjacent residuals tend to take
on similar values.  Finally, the center panel illustrates a more moderate case
in which the residuals had a correlation of 0.5.  There is still evidence of
tracking, but the pattern is less pronounced. 

![img](figures/fig3_10.png "Plots of residuals from simulated time series data sets generated with differeing levels of correlation \(\rho\) between error terms for adjacent time points.")

In the left-hand panel of figure [77](#org83e9a2b), the magnitude of the
residuals tends to increase with the fitted values.  The right hand panel
displays residual plot after transforming the response using \(\log(Y)\).  The
residuals now appear to have constant variance, although there is some evidence
of a non-linear relationship in the data.

![img](figures/fig3_11.png "Residual plots.  The red line, a smooth fit to the residuals, is intended to make it easier to identify a trend.  The blue lines track \(5^{th}\) and \(95^{th}\) percentiles of the residuals, and emphasize patterns.  Left: The funnel shape indicates heteroscedasticity.  Right: the response has been log transformed, and now there is no evidence of heteroscedasticity.")

The red point (observation 20) in the left hand panel of figure
[79](#org4f4b52f) illustrates a typical outlier.  The red solid line is the
least squares regression fit, while the blue dashed line is the least squares
fit after removal of the outlier.  In this case, removal of outlier has little
effect on the least squares line.  In the center panel of figure
[79](#org4f4b52f), the outlier is clearly visible.  In practice, to decide if
the outlier is sufficiently big to be considered an outlier, we can plot
*studentized residuals*, computed by dividing each residual \(\epsilon_i\) by its
estimated standard error.  These are shown in the right hand panel. 

![img](figures/fig3_12.png "Left: The least squares regression line is shown in red.  The regression line after removing the outlier is is shown in blue.  Center: The residual plot clearly identifies the outlier.  Right: The outlier has a studentized residual of 6; typically we expect values between -3 and 3.")

Observation 41 in the left-hand panel in figure [81](#org0dcfd6b) has
high leverage, in that the predictor value for this observation is large
relative to the other observations.  The data displayed in figure
[81](#org0dcfd6b) are the same as the data displayed in figure
[79](#org4f4b52f), except for the addition of a single high leverage
observation<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>.  The red solid line is the least squares fit to the data,
while the blue dashed line is the fit produced when observation 41 is
removed. Comparing the left-hand panels of figures [79](#org4f4b52f) and
[81](#org0dcfd6b), we observe that removing the high leverage observation has a
much more substantial impact on least squares line than removing the outlier.
The center panel of figure [81](#org0dcfd6b), for a data set with two
predictors \(X_1\) and \(X_2\). While most of the observations' predictor values
fall within the region of blue dashed lines, the red observation is well outside this
range. But neither the value for \(X_1\) nor the value for \(X_2\) is unusual.  So
if we examine just \(X_1\) or \(X_2\), we will not notice this high leverage
point. The right-panel of figure [81](#org0dcfd6b) provides a plot of
studentized residuals versus \(h_i\) for the data in the left hand panel.
Observation 41 stands out as having a very high leverage statistic as well as a
high studentized residual.

![img](figures/fig3_13.png "Left: Observation 41 is a high leverage point, while 20 is not.  The red line is the fit to all the data, and the blue line is the fit with observation 41 removed.  Center: The red observation is not unusual in terms of its \(X_1\) value or its \(X_2\) value, but still falls outside the bulk of the data, and hence has high leverage.  Right: Observation 41 has a high leverage and a high residual.")

Figure [83](#org2acb88b) illustrates the concept of collinearity.

![img](figures/fig3_14.png "Scatter plots of the observations from the `Credit` data set.  Left: A plot of `age` versus `limit`.  These two variables not collinear.  Right: A plot of `rating` versus `limit`.  There is high collinearity.")

Figure [86](#org80e1b2a) illustrates some of the difficulties that can result
from collinearity.  The left panel is a contour plot of the RSS associated with
different possible coefficient estimates for the regression of `balance` on
`limit` and `age`.  Each ellipse represents a set of coefficients that
correspond to the same RSS, with ellipses nearest to the center taking on the
lowest values of RSS.  The black dot and the associated dashed lines represent
the coefficient estimates that result in the smallest possible RSS.  The axes
for `limit` and `age` have been scaled so that the plot includes possible
coefficients that are up to four standard errors on either side of the least
squares estimates.  We see that the true `limit` coefficient is almost certainly
between 0.15 and 0.20.

In contrast, the right hand panel of figure [86](#org80e1b2a) displays contour
plots of the RSS associated with possible coefficient estimates for the
regression of `balance` onto `limit` and `rating`, which we know to be highly
collinear.  Now the contours run along a narrow valley; there is a broad range
of values for the coefficient estimates that result in equal values for RSS.  

![img](figures/fig3_15.png "Contour plots for the RSS values as a function of the parameters \(\beta\) for various regressions involving the `Credit` data set.  In each plot, the black dots represent the coefficient values corresponding to the minimum RSS.  Left: A contour plot of RSS for the regression of `balance` onto `age` and `limit`.  The minimum value is well defined.  Right: A contour plot of RSS for the regression of `balance` onto `rating` and `limit`.  Because of the collinearity, there are many pairs \((\beta_{Limit}, \beta_{Rating})\) with a similar value for RSS.")

Table [11](#org9604570) compares the coefficient estimates obtained from two
separate multiple regression models.  The first is a regression of `balance` on
`age` and `limit`.  The second is a regression of `balance` on `rating` and
`limit`.  In the first regression, both `age` and `limit` are highly significant
with very small p-values.  In the second, the collinearity between `limit` and
`rating` has caused the standard error for the `limit` coefficient to increase
by a factor of 12 and the p-value to increase to 0.701. In other words, the
importance of the `limit` variable has been masked due to the presence of
collinearity.  

<table id="org9604570" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 11:</span> The results for two multiple regression models involving the `Credit` data set.  The top panel is a regression of `balance` on `age` and `limit`.  The bottom panel is a regression of `balance` on `rating` and `limit`.  The standard error of \(\hat{\beta}_{Limit}\) increases 12-fold in the second regression, due to collinearity.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(t\)</th>
<th scope="col" class="org-right">\(P > \mid t \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">-173.411</td>
<td class="org-right">43.828</td>
<td class="org-right">-3.957</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">Age</td>
<td class="org-right">-2.291</td>
<td class="org-right">0.672</td>
<td class="org-right">-3.407</td>
<td class="org-right">0.001</td>
</tr>


<tr>
<td class="org-left">Limit</td>
<td class="org-right">0.173</td>
<td class="org-right">0.005</td>
<td class="org-right">34.496</td>
<td class="org-right">0.0</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">-377.537</td>
<td class="org-right">45.254</td>
<td class="org-right">-8.343</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">Rating</td>
<td class="org-right">2.202</td>
<td class="org-right">0.952</td>
<td class="org-right">2.312</td>
<td class="org-right">0.021</td>
</tr>


<tr>
<td class="org-left">Limit</td>
<td class="org-right">0.025</td>
<td class="org-right">0.064</td>
<td class="org-right">0.384</td>
<td class="org-right">0.701</td>
</tr>
</tbody>
</table>


<a id="org795e061"></a>

## The Marketing Plan


<a id="org133cfd7"></a>

## Comparison of Linear Regression with K-Nearest Neighbors

Figure [89](#org25f0286) illustrates two KNN fits on a data set with \(p = 2\)
predictors. The fit with \(K = 1\) is shown in the left-hand panel, while the
right-hand panel displays the fit with \(K = 9\).  When \(K = 1\), the KNN fit
perfectly interpolates the training observations, and consequently takes the
form of a step function. When \(K = 9\), the KNN fit is still a step function, but
averaging over nine observations results in much smaller regions of constant
prediction, and consequently a smoother fit.  

![img](figures/fig3_16.png "Plots of \(\hat{f}(X)\) using KNN regression on two-dimensional data set with 64 observations (brown dots).  Left: \(K = 1\) results in a rough step function fit.  Right: \(K = 9\) produces a much smoother fit.")

Figure [91](#orgfc05b2d) provides an example of KNN regression with data
generated from a one-dimensional regression model.  the black dashed lines
represent \(f(X)\), while the blue curves correspond to the KNN fits using \(K = 1\)
and \(K = 9\).  In this case, the \(K = 1\) predictions are far too variable, while
the smoother \(K = 9\) fit is much closer to \(f(X)\).  

![img](figures/fig3_17.png "Plots of \(\hat{f}(X)\) using KNN regression on a one-dimensional data set with 50 observations.  The true relationship is given by the black dashed line.  Left: The blue curve corresponds to \(K = 1\) and interpolates (i.e., passes directly through) training data.  Right: The blue curve corresponds to \(K = 9\), and represents a smoother fit.")

Figure [93](#org76084e8) represents the linear regression fit to the same
data.  It is almost perfect.  The right hand panel of figure [93](#org76084e8)
reveals that linear regression outperforms KNN for this data.  The green line,
plotted as a function of \(\frac{1}{K}\), represents the test set mean squared
error (MSE) for KNN.  The KNN errors are well above the horizontal dashed line,
which is the test MSE for linear regression.

![img](figures/fig3_18.png "The same data set shown in figure [91](#orgfc05b2d) is investigated further.  Left: The blue dashed line is the least squares fit to the data.  Since \(f(X)\) is in fact linear (displayed in black line), the least squares regression line provides a very good estimate of \(f(X)\).  Right: The dashed horizontal line represents the least squares test set MSE, while the green line corresponds to the MSE for KNN as a function of \(\frac{1}{K}\).  Linear regression achieves a lower test MSE than does KNN regression, since \(f(X)\) is in fact linear.")

Figure [95](#orgdc04fb6) examines the relative performances of least squares
regression and KNN under increasing levels of non-linearity in the relationship
between \(X\) and \(Y\).  In the top row, the true relationship is nearly linear.
In this case, we see that the test MSE for linear regression is still superior
to that of KNN for low values of \(K\) (far right).  However, as \(K\) increases,
KNN outperforms linear regression.  The second row illustrates a more
substantial deviation from linearity.  In this situation, KNN substantially
outperforms linear regression for all values of \(K\).  

![img](figures/fig3_19.png "Top Left: In a setting with a slightly non-linear relationship between \(X\) and \(Y\) (solid black line), the KNN fits with \(K = 1\) (blue) and \(K = 9\) (red) are displayed.  Top Right: For the slightly non-linear data,the test set MSE for least squares regression (horizontal) and KNN with various values of \(\frac{1}{K}\) (green) are displayed.  Bottom Left and Bottom Right: As in the top panel, but with a strongly non-linear relationship between \(X\) and \(Y\).")

Figure [97](#orgb1a56ea) considers the same strongly non-linear situation as in the lower
panel of figure [95](#orgdc04fb6), except that we have added additional *noise*
predictors that are not associated with the response.  When \(p = 1\) or \(p = 2\),
KNN outperforms linear regression.  But as we increase \(p\), linear regression
becomes superior to KNN.  In fact, increase in dimensionality has only caused a
small increase in linear regression test set MSE, but it has caused a much
bigger increase in the MSE for KNN.

![img](figures/fig3_20.png "Test MSE for linear regressions (black horizontal lines) and KNN (green curves) as the number of variables \(p\) increases.  The true function is non-linear in the first variable, as in the lower panel in figure [95](#orgdc04fb6), and does not depend upon the additional variables. The performance of linear regression deteriorates slowly in the presense of these additional variables, whereas KNN's performance degrades more quickly as \(p\) increases.")


<a id="org2af7f1e"></a>

## Lab: Linear Regression


<a id="org5ea7a3c"></a>

### Libraries

The `import` function, along with an optional `as`, is used to load *libraries*.
Before a library can be loaded, it must be installed on the system. 

    import numpy as np
    import statsmodels.formula.api as smf


<a id="orgf31ef28"></a>

### Simple Linear Regression

We load `Boston` data set from `R` library `MASS`.  Then we use `ols` function
from `statsmodels.formula.api` to fit simple linear regression model, with
`medv` as response and `lstat` as the predictor.

Function `summary2()` gives some basic information about the model.  We can use
`dir()` to find out what other pieces of information are stored in `lm_fit`.
The `predict()` function can be used to produce prediction of `medv` for a given
value of `lstat`. 

    import statsmodels.formula.api as smf
    from statsmodels import datasets
    
    boston = datasets.get_rdataset('Boston', 'MASS').data
    print(boston.columns)
    print('--------')
    
    lm_reg = smf.ols(formula='medv ~ lstat', data=boston)
    lm_fit = lm_reg.fit()
    print(lm_fit.summary2())
    print('------')
    
    print(dir(lm_fit))
    print('------')
    
    print(lm_fit.predict(exog=dict(lstat=[5, 10, 15])))

    Index(['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
           'ptratio', 'black', 'lstat', 'medv'],
          dtype='object')
    --------
                     Results: Ordinary least squares
    ==================================================================
    Model:              OLS              Adj. R-squared:     0.543    
    Dependent Variable: medv             AIC:                3286.9750
    Date:               2019-05-28 14:10 BIC:                3295.4280
    No. Observations:   506              Log-Likelihood:     -1641.5  
    Df Model:           1                F-statistic:        601.6    
    Df Residuals:       504              Prob (F-statistic): 5.08e-88 
    R-squared:          0.544            Scale:              38.636   
    -------------------------------------------------------------------
                   Coef.   Std.Err.     t      P>|t|    [0.025   0.975]
    -------------------------------------------------------------------
    Intercept     34.5538    0.5626   61.4151  0.0000  33.4485  35.6592
    lstat         -0.9500    0.0387  -24.5279  0.0000  -1.0261  -0.8740
    ------------------------------------------------------------------
    Omnibus:             137.043       Durbin-Watson:          0.892  
    Prob(Omnibus):       0.000         Jarque-Bera (JB):       291.373
    Skew:                1.453         Prob(JB):               0.000  
    Kurtosis:            5.319         Condition No.:          30     
    ==================================================================
    
    ------
    ['HC0_se', 'HC1_se', 'HC2_se', 'HC3_se', '_HCCM', '__class__', '__delattr__', 
    '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', 
    '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', 
    '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', 
    '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', 
    '__subclasshook__', '__weakref__', '_cache', '_data_attr', 
    '_get_robustcov_results', '_is_nested', '_wexog_singular_values', 'aic', 
    'bic', 'bse', 'centered_tss', 'compare_f_test', 'compare_lm_test', 
    'compare_lr_test', 'condition_number', 'conf_int', 'conf_int_el', 'cov_HC0', 
    'cov_HC1', 'cov_HC2', 'cov_HC3', 'cov_kwds', 'cov_params', 'cov_type', 
    'df_model', 'df_resid', 'eigenvals', 'el_test', 'ess', 'f_pvalue', 'f_test', 
    'fittedvalues', 'fvalue', 'get_influence', 'get_prediction', 
    'get_robustcov_results', 'initialize', 'k_constant', 'llf', 'load', 'model', 
    'mse_model', 'mse_resid', 'mse_total', 'nobs', 'normalized_cov_params', 
    'outlier_test', 'params', 'predict', 'pvalues', 'remove_data', 'resid', 
    'resid_pearson', 'rsquared', 'rsquared_adj', 'save', 'scale', 'ssr', 
    'summary', 'summary2', 't_test', 't_test_pairwise', 'tvalues', 
    'uncentered_tss', 'use_t', 'wald_test', 'wald_test_terms', 'wresid']
    ------
    0    29.803594
    1    25.053347
    2    20.303101
    dtype: float64

We will now plot `medv` and `lstat` along with least squares regression line.

    import statsmodels.formula.api as smf
    from statsmodels import datasets
    
    boston = datasets.get_rdataset('Boston', 'MASS').data
    print(boston.columns)
    print('--------')
    
    lm_reg = smf.ols(formula='medv ~ lstat', data=boston)
    lm_fit = lm_reg.fit()
    print(lm_fit.summary2())
    print('------')
    
    print(dir(lm_fit))
    print('------')
    
    print(lm_fit.predict(exog=dict(lstat=[5, 10, 15])))
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    boston.plot(x='lstat', y='medv', alpha=0.7, ax=ax)
    sm.graphics.abline_plot(model_results=lm_fit, ax=ax, c='r')

Next we examine some diagnostic plots.  

    import statsmodels.formula.api as smf
    from statsmodels import datasets
    
    boston = datasets.get_rdataset('Boston', 'MASS').data
    print(boston.columns)
    print('--------')
    
    lm_reg = smf.ols(formula='medv ~ lstat', data=boston)
    lm_fit = lm_reg.fit()
    print(lm_fit.summary2())
    print('------')
    
    print(dir(lm_fit))
    print('------')
    
    print(lm_fit.predict(exog=dict(lstat=[5, 10, 15])))
    import statsmodels.api as sm
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(lm_fit.fittedvalues, lm_fit.resid, s=5, c='b', alpha=0.6)
    ax1.axhline(y=0, linestyle='--', c='r')
    # resid_lowess_fit = lowess(endog=lm_fit.resid, exog=lm_fit.fittedvalues,
    #                           is_sorted=True)
    # ax1.plot(resid_lowess_fit[:,0], resid_lowess_fit[:,1]) 
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    
    ax2=fig.add_subplot(222)
    sm.graphics.qqplot(lm_fit.resid, ax=ax2, markersize=3, line='s',
    		   linestyle='--', fit=True, alpha=0.4)
    ax2.set_ylabel('Standardized residuals')
    ax2.set_title('Normal Q-Q')
    
    influence = lm_fit.get_influence()
    standardized_resid = influence.resid_studentized_internal
    ax3 = fig.add_subplot(223)
    ax3.scatter(lm_fit.fittedvalues, np.sqrt(np.abs(standardized_resid)), s=5,
    	    alpha=0.4, c='b')
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel(r'$\sqrt{\mid Standardized\; residuals \mid}$')
    ax3.set_title('Scale-Location')
    
    ax4 = fig.add_subplot(224)
    sm.graphics.influence_plot(lm_fit, size=2, alpha=0.4, c='b',  ax=ax4)
    ax4.xaxis.label.set_size(10)
    ax4.yaxis.label.set_size(10)
    ax4.title.set_size(12)
    ax4.set_xlim(0, 0.03)
    for txt in ax4.texts:
        txt.set_visible(False)
    ax4.axhline(y=0, linestyle='--', color='grey')
    
    fig.tight_layout()


<a id="org4d8f496"></a>

### Multiple Linear Regression

In order to fit a multiple regression model using least squares, we again use
the `ols` and `fit` functions.  The syntax `ols(formula='y ~ x1 + x2 + x3')` is
used to fit a model with three predictors, `x1`, `x2`, and `x3`.  The
`summary2()` now outputs the regression coefficients for all three predictors. 

`statsmodels` does not seem to have `R` like facility to include all variables
using the formula `y ~ .`.  To include all variables, we either write them
individually, or use code to create a formula.

    import statsmodels.formula.api as smf
    from statsmodels import datasets
    
    boston = datasets.get_rdataset('Boston', 'MASS').data
    
    lm_reg = smf.ols(formula='medv ~ lstat + age', data=boston)
    lm_fit = lm_reg.fit()
    
    print(lm_fit.summary2())
    print('--------')
    
    # Create formula to include all variables
    all_columns = list(boston.columns)
    all_columns.remove('medv')
    my_formula = 'medv ~ ' + ' + '.join(all_columns)
    print(my_formula)
    print('--------')
    
    all_reg = smf.ols(formula=my_formula, data=boston)
    all_fit = all_reg.fit()
    print(all_fit.summary2())
    print('--------')

                     Results: Ordinary least squares
    ==================================================================
    Model:              OLS              Adj. R-squared:     0.549    
    Dependent Variable: medv             AIC:                3281.0064
    Date:               2019-05-29 10:07 BIC:                3293.6860
    No. Observations:   506              Log-Likelihood:     -1637.5  
    Df Model:           2                F-statistic:        309.0    
    Df Residuals:       503              Prob (F-statistic): 2.98e-88 
    R-squared:          0.551            Scale:              38.108   
    -------------------------------------------------------------------
                   Coef.   Std.Err.     t      P>|t|    [0.025   0.975]
    -------------------------------------------------------------------
    Intercept     33.2228    0.7308   45.4579  0.0000  31.7869  34.6586
    lstat         -1.0321    0.0482  -21.4163  0.0000  -1.1267  -0.9374
    age            0.0345    0.0122    2.8256  0.0049   0.0105   0.0586
    ------------------------------------------------------------------
    Omnibus:             124.288       Durbin-Watson:          0.945  
    Prob(Omnibus):       0.000         Jarque-Bera (JB):       244.026
    Skew:                1.362         Prob(JB):               0.000  
    Kurtosis:            5.038         Condition No.:          201    
    ==================================================================
    
    --------
    medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + 
    ptratio + black + lstat
    --------
                     Results: Ordinary least squares
    ==================================================================
    Model:              OLS              Adj. R-squared:     0.734    
    Dependent Variable: medv             AIC:                3025.6086
    Date:               2019-05-29 10:07 BIC:                3084.7801
    No. Observations:   506              Log-Likelihood:     -1498.8  
    Df Model:           13               F-statistic:        108.1    
    Df Residuals:       492              Prob (F-statistic): 6.72e-135
    R-squared:          0.741            Scale:              22.518   
    -------------------------------------------------------------------
                Coef.    Std.Err.     t      P>|t|    [0.025    0.975] 
    -------------------------------------------------------------------
    Intercept   36.4595    5.1035    7.1441  0.0000   26.4322   46.4868
    crim        -0.1080    0.0329   -3.2865  0.0011   -0.1726   -0.0434
    zn           0.0464    0.0137    3.3816  0.0008    0.0194    0.0734
    indus        0.0206    0.0615    0.3343  0.7383   -0.1003    0.1414
    chas         2.6867    0.8616    3.1184  0.0019    0.9939    4.3796
    nox        -17.7666    3.8197   -4.6513  0.0000  -25.2716  -10.2616
    rm           3.8099    0.4179    9.1161  0.0000    2.9887    4.6310
    age          0.0007    0.0132    0.0524  0.9582   -0.0253    0.0266
    dis         -1.4756    0.1995   -7.3980  0.0000   -1.8675   -1.0837
    rad          0.3060    0.0663    4.6129  0.0000    0.1757    0.4364
    tax         -0.0123    0.0038   -3.2800  0.0011   -0.0197   -0.0049
    ptratio     -0.9527    0.1308   -7.2825  0.0000   -1.2098   -0.6957
    black        0.0093    0.0027    3.4668  0.0006    0.0040    0.0146
    lstat       -0.5248    0.0507  -10.3471  0.0000   -0.6244   -0.4251
    ------------------------------------------------------------------
    Omnibus:             178.041       Durbin-Watson:          1.078  
    Prob(Omnibus):       0.000         Jarque-Bera (JB):       783.126
    Skew:                1.521         Prob(JB):               0.000  
    Kurtosis:            8.281         Condition No.:          15114  
    ==================================================================
    * The condition number is large (2e+04). This might indicate
    strong multicollinearity or other numerical problems.
    --------


<a id="orgae8db59"></a>

### Interaction Terms

The syntax `lstat:black` tells `ols` to include an interaction term between
`lstat` and `black`.  The syntax `lstat*age` simultaneously includes `lstat,
age,` and the interaction term \(\text{lstat} \times \text{age]\) as predictors.
It is a shorthand for `lstat + age + lstat:age`. 

    import statsmodels.formula.api as smf
    from statsmodels import datasets
    
    boston = datasets.get_rdataset('Boston', 'MASS').data
    
    my_reg = smf.ols(formula='medv ~ lstat * age', data=boston)
    my_fit = my_reg.fit()
    print(my_fit.summary2())

                     Results: Ordinary least squares
    ==================================================================
    Model:              OLS              Adj. R-squared:     0.553    
    Dependent Variable: medv             AIC:                3277.9547
    Date:               2019-05-29 11:48 BIC:                3294.8609
    No. Observations:   506              Log-Likelihood:     -1635.0  
    Df Model:           3                F-statistic:        209.3    
    Df Residuals:       502              Prob (F-statistic): 4.86e-88 
    R-squared:          0.556            Scale:              37.804   
    -------------------------------------------------------------------
                    Coef.   Std.Err.     t     P>|t|    [0.025   0.975]
    -------------------------------------------------------------------
    Intercept      36.0885    1.4698  24.5528  0.0000  33.2007  38.9763
    lstat          -1.3921    0.1675  -8.3134  0.0000  -1.7211  -1.0631
    age            -0.0007    0.0199  -0.0363  0.9711  -0.0398   0.0383
    lstat:age       0.0042    0.0019   2.2443  0.0252   0.0005   0.0078
    ------------------------------------------------------------------
    Omnibus:             135.601       Durbin-Watson:          0.965  
    Prob(Omnibus):       0.000         Jarque-Bera (JB):       296.955
    Skew:                1.417         Prob(JB):               0.000  
    Kurtosis:            5.461         Condition No.:          6878   
    ==================================================================
    * The condition number is large (7e+03). This might indicate
    strong multicollinearity or other numerical problems.


<a id="org166f20d"></a>

### Non-linear Transformations of the Predictors

The `ols` function can also accommodate non-linear transformations of the
predictors.  For example, given a predictor \(X\), we can create predictor \(X^2\)
using `I(X ** 2)`.  We now perform a regression of `medv` onto `lstat` and
\(\texttt{lstat}^2\). 

The near-zero p-value associated with the quadratic term suggests that it leads
to an improve model.  We use `anova_lm()` function to further quantify the
extent to which the quadratic fit is superior to the linear fit.  The null
hypothesis is that the two models fit the data equally well.  The alternative
hypothesis is that the full model is superior.  Given the large F-statistic and
zero p-value, this provides very clear evidence that the model with quadratic
term is superior.  A plot of residuals versus fitted values shows that, with
quadratic term included, there is no discernible pattern in residuals. 

    import statsmodels.formula.api as smf
    from statsmodels import datasets
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    import matplotlib.pyplot as plt
    
    boston = datasets.get_rdataset('Boston', 'MASS').data
    
    my_reg = smf.ols(formula='medv ~ lstat', data=boston)
    my_fit = my_reg.fit()
    
    my_reg2 = smf.ols(formula='medv ~ lstat + I(lstat ** 2)', data=boston)
    my_fit2 = my_reg2.fit()
    print(my_fit.summary2())
    print('--------')
    
    print(sm.stats.anova_lm(my_fit2))
    print('--------')
    
    print(sm.stats.anova_lm(my_fit, my_fit2))
    
    my_regs = (my_reg, my_reg2)
    
    fig = plt.figure(figsize=(8,4))
    i_reg = 1
    for reg in my_regs:
        ax = fig.add_subplot(1, 2, i_reg)
        fit = reg.fit()
        ax.scatter(fit.fittedvalues, fit.resid, s=7, alpha=0.6)
        lowess_fit = lowess(fit.resid, fit.fittedvalues)
        ax.plot(lowess_fit[:,0], lowess_fit[:,1], c='r')
        ax.axhline(y=0, linestyle='--', color='grey')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        ax.set_title(reg.formula)
        i_reg += 1
    
    fig.tight_layout()

                     Results: Ordinary least squares
    ==================================================================
    Model:              OLS              Adj. R-squared:     0.543    
    Dependent Variable: medv             AIC:                3286.9750
    Date:               2019-05-29 12:41 BIC:                3295.4280
    No. Observations:   506              Log-Likelihood:     -1641.5  
    Df Model:           1                F-statistic:        601.6    
    Df Residuals:       504              Prob (F-statistic): 5.08e-88 
    R-squared:          0.544            Scale:              38.636   
    -------------------------------------------------------------------
                   Coef.   Std.Err.     t      P>|t|    [0.025   0.975]
    -------------------------------------------------------------------
    Intercept     34.5538    0.5626   61.4151  0.0000  33.4485  35.6592
    lstat         -0.9500    0.0387  -24.5279  0.0000  -1.0261  -0.8740
    ------------------------------------------------------------------
    Omnibus:             137.043       Durbin-Watson:          0.892  
    Prob(Omnibus):       0.000         Jarque-Bera (JB):       291.373
    Skew:                1.453         Prob(JB):               0.000  
    Kurtosis:            5.319         Condition No.:          30     
    ==================================================================
    
    --------
                      df        sum_sq       mean_sq           F         PR(>F)
    lstat            1.0  23243.913997  23243.913997  761.810354  8.819026e-103
    I(lstat ** 2)    1.0   4125.138260   4125.138260  135.199822   7.630116e-28
    Residual       503.0  15347.243158     30.511418         NaN            NaN
    --------
       df_resid           ssr  df_diff     ss_diff           F        Pr(>F)
    0     504.0  19472.381418      0.0         NaN         NaN           NaN
    1     503.0  15347.243158      1.0  4125.13826  135.199822  7.630116e-28


<a id="orga3ed622"></a>

### Qualitative Predictors

We will now examine `Carseats` data, which is part of the `ISLR` library.  We
will attempt to predict `Sales` (child car seat sales) based on a number of
predictors. `statsmodels` automatically converts string variables into
categorical variables.  If we want `statsmodels` to treat a numerical variable `x` as
qualitative predictor, the formula should be `y ~ C(x)`. Here `C()` stands for
categorical.  

    import statsmodels.formula.api as smf
    from statsmodels import datasets
    
    carseats = datasets.get_rdataset('Carseats', 'ISLR').data
    print(carseats.columns)
    print('--------')
    
    all_columns = list(carseats.columns)
    all_columns.remove('Sales')
    my_formula = 'Sales ~ ' + ' + '.join(all_columns)
    my_formula +=  ' + Income:Advertising + Price:Age'
    
    print(my_formula)
    print('--------')
    
    my_reg = smf.ols(formula=my_formula, data=carseats)
    my_fit = my_reg.fit()
    print(my_fit.summary2())

    Index(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
           'ShelveLoc', 'Age', 'Education', 'Urban', 'US'],
          dtype='object')
    --------
    Sales ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc + 
    Age + Education + Urban + US + Income:Advertising + Price:Age
    --------
                      Results: Ordinary least squares
    ====================================================================
    Model:                OLS              Adj. R-squared:     0.872    
    Dependent Variable:   Sales            AIC:                1157.3378
    Date:                 2019-05-29 12:53 BIC:                1213.2183
    No. Observations:     400              Log-Likelihood:     -564.67  
    Df Model:             13               F-statistic:        210.0    
    Df Residuals:         386              Prob (F-statistic): 6.14e-166
    R-squared:            0.876            Scale:              1.0213   
    --------------------------------------------------------------------
                         Coef.  Std.Err.    t     P>|t|   [0.025  0.975]
    --------------------------------------------------------------------
    Intercept            6.5756   1.0087   6.5185 0.0000  4.5922  8.5589
    ShelveLoc[T.Good]    4.8487   0.1528  31.7243 0.0000  4.5482  5.1492
    ShelveLoc[T.Medium]  1.9533   0.1258  15.5307 0.0000  1.7060  2.2005
    Urban[T.Yes]         0.1402   0.1124   1.2470 0.2132 -0.0808  0.3612
    US[T.Yes]           -0.1576   0.1489  -1.0580 0.2907 -0.4504  0.1352
    CompPrice            0.0929   0.0041  22.5668 0.0000  0.0848  0.1010
    Income               0.0109   0.0026   4.1828 0.0000  0.0058  0.0160
    Advertising          0.0702   0.0226   3.1070 0.0020  0.0258  0.1147
    Population           0.0002   0.0004   0.4329 0.6653 -0.0006  0.0009
    Price               -0.1008   0.0074 -13.5494 0.0000 -0.1154 -0.0862
    Age                 -0.0579   0.0160  -3.6329 0.0003 -0.0893 -0.0266
    Education           -0.0209   0.0196  -1.0632 0.2884 -0.0594  0.0177
    Income:Advertising   0.0008   0.0003   2.6976 0.0073  0.0002  0.0013
    Price:Age            0.0001   0.0001   0.8007 0.4238 -0.0002  0.0004
    --------------------------------------------------------------------
    Omnibus:                1.281        Durbin-Watson:           2.047 
    Prob(Omnibus):          0.527        Jarque-Bera (JB):        1.147 
    Skew:                   0.129        Prob(JB):                0.564 
    Kurtosis:               3.050        Condition No.:           130576
    ====================================================================
    * The condition number is large (1e+05). This might indicate
    strong multicollinearity or other numerical problems.


<a id="orgcddf23b"></a>

### Calling `R` from `Python`


<a id="org0943dd8"></a>

# Classification


<a id="orgb649a04"></a>

## An Overview of Classification

In figure [110](#org6935596), we have plotted annual `income` and monthly
credit card `balance` for a subset of individuals in `Credit` data set.  The
left hand panel displays individuals who defaulted in brown, and those who did
not in blue.  We have plotted only a fraction of individuals who did not
default.  It appears that individuals who defaulted tended to have higher credit
card balances than those who did not.  In the right hand panel, we show two
pairs of boxplots.  The first shows the distribution of `balance` split by the
binary `default` variable; the second is a similar plot for `income`.  

![img](figures/fig4_1.png "The `Default` data set.  Left: The annual income and monthly credit card balances of a number of individuals.  The individuals who defaulted on their credit card debt are shown in brown, and those who did not default are shown in blue.  Center: Boxplots of `balance` as a function of `default` status.  Right: Boxplots of `income` as a function of `default` status.")


<a id="org8afbfa7"></a>

## Why Not Linear Regression?


<a id="orgbbdfc0d"></a>

## Logistic Regression

Using `Default` data set, in figure [112](#org78c765a) we show probability of default as a function of
`balance`.  The left panel shows a model fitted using linear regression.  Some
of the probabilities estimates (for low balance) are outside the \([0, 1]\)
interval.  The right panel shows a model fitted using logistic regression, which
models the probability of default as a function of `balance`.  Now all
probability estimates are in the \([0, 1]\) interval.

![img](figures/fig4_2.png "Classification using `Default` data.  Left: Estimated probability of `default` using linear regression.  Some estimated probabilities are negative!  The brown ticks indicate the 0/1 values coded for `default` (`No` or `Yes`).  Right: Predicted probabilities of `default` using logistic regression.  All probabilities lie between 0 and 1.")

Table [12](#org85e7f6c) shows the coefficient estimates and related
information that result from fitting a logistic regression model on the
`Default` data in order to predict the probability of `default = Yes` using `balance`.

<table id="org85e7f6c" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 12:</span> For the `Default` data, estimated coefficients of the logistic regression model that predicts the probability of `default` using `balance`.  A one-unit increase in `balance` is associated with an increase in the log odds of `default` by 0.0055 units.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(z\)</th>
<th scope="col" class="org-right">\(P > \mid z \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">-10.6513</td>
<td class="org-right">0.3612</td>
<td class="org-right">-29.4913</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">balance</td>
<td class="org-right">0.0055</td>
<td class="org-right">0.0002</td>
<td class="org-right">24.9524</td>
<td class="org-right">0.0</td>
</tr>
</tbody>
</table>

Table [13](#orge050a44) shows the results of logistic model where `default`
is a function of the qualitative variable `student`.  

Table [14](#org606b34b) shows the coefficient estimates for a logistic
regression model that uses `balance`, `income` (in thousands of dollars), and
`student` status to predict probability of `default`.

<table id="orge050a44" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 13:</span> For the `Default` data, estimated coefficients of the logistic regression model that predicts the probability of `default` using student status.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(z\)</th>
<th scope="col" class="org-right">\(P > \mid z \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">-3.5041</td>
<td class="org-right">0.0707</td>
<td class="org-right">-49.5541</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">student[T.Yes]</td>
<td class="org-right">0.4049</td>
<td class="org-right">0.115</td>
<td class="org-right">3.5202</td>
<td class="org-right">0.0004</td>
</tr>
</tbody>
</table>

<table id="org606b34b" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 14:</span> For the `Default` data, estimated coefficients of the logistic regression model that predicts the probability of `default` using `balance`, `income`, and `student` status.  In fitting this model, `income` was measured in thousands of dollars.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">Coef.</th>
<th scope="col" class="org-right">Std.Err.</th>
<th scope="col" class="org-right">\(z\)</th>
<th scope="col" class="org-right">\(P > \mid z \mid\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Intercept</td>
<td class="org-right">-10.869</td>
<td class="org-right">0.4923</td>
<td class="org-right">-22.0793</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">student[T.Yes]</td>
<td class="org-right">-0.6468</td>
<td class="org-right">0.2363</td>
<td class="org-right">-2.7376</td>
<td class="org-right">0.0062</td>
</tr>


<tr>
<td class="org-left">balance</td>
<td class="org-right">0.0057</td>
<td class="org-right">0.0002</td>
<td class="org-right">24.7365</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">income</td>
<td class="org-right">0.003</td>
<td class="org-right">0.0082</td>
<td class="org-right">0.3698</td>
<td class="org-right">0.7115</td>
</tr>
</tbody>
</table>

The left hand panel of figure [117](#org0b960da) shows average default rates
for students and non-students, respectively, as a function of credit card
balance.  *For a fixed value* of `balance` and `income`, a student is less
likely to default than a non-student.  This is true for all values of balance.
This is consistent with negative coefficient of student in table
[14](#org606b34b).  But the horizontal lines near the base of the plot, which show the default rates
for students and non-students averaged over all values of `balance` and
`income`, suggest the opposite effect: the overall student default rate is
higher than non-student default rate.  Consequently, there is a positive
coefficient for `student` in the single variable logistic regression output
shown in table [13](#orge050a44). 

![img](figures/fig4_3.png "Confounding in the `Default` data.  Left: Default rates are shown for students (brown) and non-students (blue).  The solid lines display default rate as a function of `balance`, while the horizontal lines display the overall default rates.  Right: Boxplots of `balance` for students and non-students are shown.")


<a id="org6a9ceec"></a>

## Linear Discriminant Analysis

In the left panel of figure [119](#org0313602), two normal density functions
that are displayed, \(f_1(x)\) and \(f_2(x)\), represent two distinct classes.  The
Bayes classifier boundary, shown as vertical dashed line, is estimated using the
function `GaussianNB()`.  The right hand panel displays a histogram of a random
sample of 20 observations from each class.  The LDA decision boundary is shown
as firm vertical line.

![img](figures/fig4_4.png "Left: Two one-dimensional normal density functions are shown.  The dashed vertical line represents the Bayes decision boundary.  Right: 20 observations were drawn from each of the two classes, and are shown as histograms.  The Bayes decision boundary is again shown as a dashed vertical line.  The solid vertical line represents the LDA decision boundary estimated from the training data.")

Two examples of multivariate Gaussian distributions with \(p = 2\) are shown in
figure [121](#org66e025a).  In the upper panel, the height of the surface at
any particular point represents the probability that both \(X_1\) and \(X_2\) fall
in the small region around that point.  If the surface is cut along the \(X_1\)
axis or along the \(X_2\) axis, the resulting cross-section will have the shape of
a one-dimensional normal distribution.  The left-hand panel illustrates an example in
which \(\text{var}(X_1) = \text{var}(X_2)\) and \(\text{cor}(X_1, X_2) = 0\); this surface has a
characteristic *bell shape*.  However, the bell shape will be distorted if the
predictors are correlated or have unequal variances, as is illustrated in the
right-hand panel of figure [121](#org66e025a).  In this situation, the base
of the bell will have an elliptical, rather than circular, shape.  The contour
plots in the lower panel are not in the book. 

![img](figures/fig4_5.png "Two multivariate Gaussian density functions are shown, with \(p = 2\).  Left: The two predictors are uncorrelated.  Right: The two predictors have a correlation of 0.7.  The lower panel shows contour plots of the surfaces drawn in the upper panel.  Here the correlations can be easily seen.")

Figure [123](#orgfb74e1c) shows an example of three equally sized Gaussian
classes with class-specific mean vectors and a common covariance matrix. The
dashed lines are the Bayes decision boundaries.  

![img](figures/fig4_6.png "An example with three classes. The observation from each class are drawn from a multivariate Gaussian distribution with \(p = 2\), with a class-specific mean vector and a common covariance matrix.  Left: The dashed lines are the Bayes decision boundaries.  Right: 20 observations were generated from each class, and the corresponding LDA decision boundaries are indicated using solid black lines.  The Bayes decision boundaries are once again shown as dashed lines.")

A *confusion matrix*, shown for the `Default` data in table
[15](#orgf8c63e1), is a convenient way to display prediction of default in
comparison to true default.  Table [16](#org2035f16) shows the error rates
that result when we label any customer with a posterior probability of default
above 20% to the *default* class.

<table id="orgf8c63e1" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 15:</span> A confusion matrix compares the LDA predictions to the true default statuses for the training observations in the `Default` data set.  Elements of the diagonal matrix represent individuals whose default statuses were correctly predicted, while off-diagonal elements represent individuals that were missclassified.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">true No</th>
<th scope="col" class="org-right">true Yes</th>
<th scope="col" class="org-right">Total</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">predict No</td>
<td class="org-right">9645</td>
<td class="org-right">254</td>
<td class="org-right">9899</td>
</tr>


<tr>
<td class="org-left">predict Yes</td>
<td class="org-right">22</td>
<td class="org-right">79</td>
<td class="org-right">101</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Total</td>
<td class="org-right">9667</td>
<td class="org-right">333</td>
<td class="org-right">10000</td>
</tr>
</tbody>
</table>

<table id="org2035f16" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 16:</span> A confusion matrix compares LDA predictions to the true default statuses for the training observations in the `Default` data set, using a modified threshold value that predicts default for any individuals whose posterior default probability exceeds 20%.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">true No</th>
<th scope="col" class="org-right">true Yes</th>
<th scope="col" class="org-right">Total</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">predict No</td>
<td class="org-right">9435</td>
<td class="org-right">140</td>
<td class="org-right">9575</td>
</tr>


<tr>
<td class="org-left">predict Yes</td>
<td class="org-right">232</td>
<td class="org-right">193</td>
<td class="org-right">425</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Total</td>
<td class="org-right">9667</td>
<td class="org-right">333</td>
<td class="org-right">10000</td>
</tr>
</tbody>
</table>

Figure [126](#orgeab0a38) illustrates the trade-off that results from
modifying the threshold value for the posterior probability of default.  Various
error rates are shown as a function of the threshold value.  Using a threshold
of 0.5 minimizes the overall error rate, shown as a black line.  But when a
threshold of 0.5 is used, the error rate among the individuals who default is
quite high (blue dashed line).  As the threshold is reduced, the error rate
among individuals who default decreases steadily, but the error rate amond
individuals who do not default increases. 

![img](figures/fig4_7.png "For the `Default` data set, error rates are shown as a function of the threshold value for the posterior probability that is used to perform the assignment of default.  The black sold line displays the overall error rate.  The blue dashed line represents the fraction of defaulting customers that are incorrectly classified, and the orange dotted line indicates the fraction of errors among the non-defaulting customers.")

Figure [128](#orgdf355a4) displays the ROC curve for the LDA classifier on
the `Default` data set.

![img](figures/fig4_8.png "A ROC curve for the LDA classifier on the `Default` data.  It traces two types of error as we vary the threshold value for the posterior probability of default.  The actual thresholds are not shown.  The true positive rate is the sensitivity: the fraction of defaulters that are correctly identified using a given threshold value.  The false positive rate is the fraction of non-defaulters we incorrectly specify as defaulters, using the same threshold value.  The ideal ROC curve hugs the top left corner, indicating a high true positive rate and a low false positive rate.  The dotted line represents the \`\`no information'' classifier; this is what we would expect if student status and credit card balance are not associated with the probability of default.")

Table [17](#org88ae016) shows the possible results when applying a
classifier (or diagnostic test) to a population.  

<table id="org88ae016" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 17:</span> Possible results when applying a classifier or diagnostic test to a population.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">*True class*</th>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">&#xa0;</th>
</tr>


<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">- or Null</th>
<th scope="col" class="org-left">+ or Non-null</th>
<th scope="col" class="org-left">Total</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">*Predicted*</td>
<td class="org-left">- or Null</td>
<td class="org-left">True Negative (TN)</td>
<td class="org-left">False Negative (FN)</td>
<td class="org-left">N\*</td>
</tr>


<tr>
<td class="org-left">*class*</td>
<td class="org-left">+ or Non-null</td>
<td class="org-left">False Positive (FP)</td>
<td class="org-left">True Positive (TP)</td>
<td class="org-left">P\*</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">&#xa0;</td>
<td class="org-left">Total</td>
<td class="org-left">N</td>
<td class="org-left">P</td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>
</table>

Table [18](#org4039a38) lists many of the popular performance measures that
are used in this context.

<table id="org4039a38" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 18:</span> Important measures for classification and diagnostic testing, derived from quantities in table [17](#org88ae016).</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Name</th>
<th scope="col" class="org-left">Definition</th>
<th scope="col" class="org-left">Synonyms</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">False Positive rate</td>
<td class="org-left">FP / N</td>
<td class="org-left">Type I error, 1 - specificity</td>
</tr>


<tr>
<td class="org-left">True Positive rate</td>
<td class="org-left">TP / P</td>
<td class="org-left">1 - Type II error, power, sensitivity, recall</td>
</tr>


<tr>
<td class="org-left">Positive Predicted value</td>
<td class="org-left">TP / P\*</td>
<td class="org-left">Precision, 1 - false discovery proportion</td>
</tr>


<tr>
<td class="org-left">Negative Predicted value</td>
<td class="org-left">TN / N\*</td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>
</table>

Figure [132](#org95a44c2) illustrates the performances of LDA and QDA in two
scenarios.  In the left-hand panel, the two Gaussian classes have a common
correlation of 0.7 between \(X_1\) and \(X_2\).  As a result, the Bayes decision
boundary is nearly linear and is accurately approximated by the LDA decision
boundary.  In contrast, the right-hand panel displays a situation in which the
orange class has a correlation of 0.7 between the variables and blue class has a
correlation of -0.7.

![img](figures/fig4_9.png "Left: The Bayes (purple dashed), LDA (black dotted), and QDA (green sold) decision boundaries for a two-class problem with \(\Sigma_1 = \Sigma_2\).  Right: Details are as given in the left-hand panel, except that \(\Sigma_1 \ne \Sigma_2\).")


<a id="orgb5841fc"></a>

## A Comparison of Classification Methods

Figure [134](#org0040d63) illustrates the performances of the four
classification approaches (KNN, LDA, Logistic, and QDA) when Bayes decision
boundary is linear.

![img](figures/fig4_10.png "Boxplots of the test error rates for each of the linear scenarios described in the main text.")


<a id="orga0e83c5"></a>

## Lab: Logistic Regression, LDA, QDA, and KNN


<a id="org34b2dd2"></a>

### The Stock Market Data

We will begin by examining some numerical and graphical summaries of the
`Smarket` data, which is part of the `ISLR` library.

    from statsmodels import datasets
    import pandas as pd
    
    smarket = datasets.get_rdataset('Smarket', 'ISLR').data
    
    print(smarket.columns)
    print('--------')
    print(smarket.shape)
    print('--------')
    print(smarket.describe())
    print('--------')
    print(smarket.iloc[:,1:8].corr())
    print('--------')
    smarket.boxplot(column='Volume', by='Year', grid=False)

    Index(['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today',
           'Direction'],
          dtype='object')
    --------
    (1250, 9)
    --------
                  Year         Lag1  ...       Volume        Today
    count  1250.000000  1250.000000  ...  1250.000000  1250.000000
    mean   2003.016000     0.003834  ...     1.478305     0.003138
    std       1.409018     1.136299  ...     0.360357     1.136334
    min    2001.000000    -4.922000  ...     0.356070    -4.922000
    25%    2002.000000    -0.639500  ...     1.257400    -0.639500
    50%    2003.000000     0.039000  ...     1.422950     0.038500
    75%    2004.000000     0.596750  ...     1.641675     0.596750
    max    2005.000000     5.733000  ...     3.152470     5.733000
    
    [8 rows x 8 columns]
    --------
                Lag1      Lag2      Lag3      Lag4      Lag5    Volume     Today
    Lag1    1.000000 -0.026294 -0.010803 -0.002986 -0.005675  0.040910 -0.026155
    Lag2   -0.026294  1.000000 -0.025897 -0.010854 -0.003558 -0.043383 -0.010250
    Lag3   -0.010803 -0.025897  1.000000 -0.024051 -0.018808 -0.041824 -0.002448
    Lag4   -0.002986 -0.010854 -0.024051  1.000000 -0.027084 -0.048414 -0.006900
    Lag5   -0.005675 -0.003558 -0.018808 -0.027084  1.000000 -0.022002 -0.034860
    Volume  0.040910 -0.043383 -0.041824 -0.048414 -0.022002  1.000000  0.014592
    Today  -0.026155 -0.010250 -0.002448 -0.006900 -0.034860  0.014592  1.000000
    --------


<a id="orgf525a1b"></a>

### Logistc Regression

Next, we will fit a logistic regression model to predict `Direction` using
`Lag1` through `Lag5` and `Volume`.

    from statsmodels import datasets
    import statsmodels.formula.api as smf
    import numpy as np
    import pandas as pd
    
    smarket = datasets.get_rdataset('Smarket', 'ISLR').data
    smarket['direction_cat'] = smarket['Direction'].apply(lambda x: int(x=='Up'))
    
    logit_model = smf.logit(
        formula='direction_cat ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume',
        data=smarket)
    logit_fit = logit_model.fit()
    
    print(logit_fit.summary2())
    print('--------')
    print(dir(logit_fit))           # see what information is available from fit
    print('--------')
    print(logit_fit.params)         # coefficients estimates
    print('--------')
    print(logit_fit.summary2().tables[1]) # coefficients estimates, std error, and z
    print('--------')
    print(logit_fit.summary2().tables[1].iloc[:,3]) # P > |z| column only
    print('--------')
    print(logit_fit.predict()[:10]) # probabilities for training data
    print('--------')
    smarket['predict_direction'] = np.vectorize(
        lambda x: 'Up' if x > 0.5 else 'Down')(logit_fit.predict())
    print(pd.crosstab(smarket['predict_direction'], smarket['Direction']))

    Optimization terminated successfully.
             Current function value: 0.691034
             Iterations 4
                             Results: Logit
    ================================================================
    Model:              Logit            Pseudo R-squared: 0.002    
    Dependent Variable: direction_cat    AIC:              1741.5841
    Date:               2019-06-06 18:56 BIC:              1777.5004
    No. Observations:   1250             Log-Likelihood:   -863.79  
    Df Model:           6                LL-Null:          -865.59  
    Df Residuals:       1243             LLR p-value:      0.73187  
    Converged:          1.0000           Scale:            1.0000   
    No. Iterations:     4.0000                                      
    -----------------------------------------------------------------
                   Coef.   Std.Err.     z     P>|z|    [0.025  0.975]
    -----------------------------------------------------------------
    Intercept     -0.1260    0.2407  -0.5234  0.6007  -0.5978  0.3458
    Lag1          -0.0731    0.0502  -1.4566  0.1452  -0.1714  0.0253
    Lag2          -0.0423    0.0501  -0.8446  0.3984  -0.1405  0.0559
    Lag3           0.0111    0.0499   0.2220  0.8243  -0.0868  0.1090
    Lag4           0.0094    0.0500   0.1873  0.8514  -0.0886  0.1073
    Lag5           0.0103    0.0495   0.2083  0.8350  -0.0867  0.1074
    Volume         0.1354    0.1584   0.8553  0.3924  -0.1749  0.4458
    ================================================================
    
    --------
    ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', 
    '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', 
    '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', 
    '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
    '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 
    '_cache', '_data_attr', '_get_endog_name', '_get_robustcov_results', 'aic', 
    'bic', 'bse', 'conf_int', 'cov_kwds', 'cov_params', 'cov_type', 'df_model', 
    'df_resid', 'f_test', 'fittedvalues', 'get_margeff', 'initialize', 
    'k_constant', 'llf', 'llnull', 'llr', 'llr_pvalue', 'load', 'mle_retvals', 
    'mle_settings', 'model', 'nobs', 'normalized_cov_params', 'params', 
    'pred_table', 'predict', 'prsquared', 'pvalues', 'remove_data', 'resid_dev', 
    'resid_generalized', 'resid_pearson', 'resid_response', 'save', 'scale', 
    'set_null_options', 'summary', 'summary2', 't_test', 't_test_pairwise', 
    'tvalues', 'use_t', 'wald_test', 'wald_test_terms']
    --------
    Intercept   -0.126000
    Lag1        -0.073074
    Lag2        -0.042301
    Lag3         0.011085
    Lag4         0.009359
    Lag5         0.010313
    Volume       0.135441
    dtype: float64
    --------
                  Coef.  Std.Err.         z     P>|z|    [0.025    0.975]
    Intercept -0.126000  0.240737 -0.523394  0.600700 -0.597836  0.345836
    Lag1      -0.073074  0.050168 -1.456583  0.145232 -0.171401  0.025254
    Lag2      -0.042301  0.050086 -0.844568  0.398352 -0.140469  0.055866
    Lag3       0.011085  0.049939  0.221974  0.824334 -0.086793  0.108963
    Lag4       0.009359  0.049974  0.187275  0.851445 -0.088589  0.107307
    Lag5       0.010313  0.049512  0.208296  0.834998 -0.086728  0.107354
    Volume     0.135441  0.158361  0.855266  0.392404 -0.174941  0.445822
    --------
    Intercept    0.600700
    Lag1         0.145232
    Lag2         0.398352
    Lag3         0.824334
    Lag4         0.851445
    Lag5         0.834998
    Volume       0.392404
    Name: P>|z|, dtype: float64
    --------
    [0.50708413 0.48146788 0.48113883 0.51522236 0.51078116 0.50695646
     0.49265087 0.50922916 0.51761353 0.48883778]
    --------
    Direction          Down   Up
    predict_direction           
    Down                145  141
    Up                  457  507

We now use data for years 2001 through 2004 to train the model, then use data
for year 2005 to test the model. 

    from statsmodels import datasets
    import statsmodels.formula.api as smf
    import pandas as pd
    import numpy as np
    
    smarket = datasets.get_rdataset('Smarket', 'ISLR').data
    smarket['direction_cat'] = smarket['Direction'].apply(lambda x:
    						      int(x == 'Up'))
    smarket_train = smarket.loc[smarket['Year'] < 2005]
    smarket_test = smarket.loc[smarket['Year'] == 2005].copy()
    
    logit_model = smf.logit(
        formula='direction_cat ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume',
        data=smarket_train)
    logit_fit = logit_model.fit()
    
    prob_up_test = logit_fit.predict(smarket_test)
    smarket_test.loc[:,'direction_predict'] = np.vectorize(
        lambda x: 'Up' if x > 0.5 else 'Down')(prob_up_test)
    
    confusion_test = \
        pd.crosstab(smarket_test['direction_predict'], smarket_test['Direction'])
    print(confusion_test)
    print('--------')
    print(np.mean(np.mean(smarket_test['direction_predict'] ==
    		      smarket_test['Direction'])))
    print('--------')
    
    # Refit logistic regression with only Lag1 and Lag2
    logit_model = smf.logit('direction_cat ~ Lag1 + Lag2', data=smarket_train)
    logit_fit = logit_model.fit()
    prob_up_test = logit_fit.predict(smarket_test)
    smarket_test['direction_pred_2var'] = np.vectorize(
        lambda x: 'Up' if x > 0.5 else 'Down')(prob_up_test)
    
    print(pd.crosstab(smarket_test['direction_pred_2var'],
    		  smarket_test['Direction']))
    print('--------')
    
    print(np.mean(smarket_test['direction_pred_2var'] == smarket_test['Direction']))
    print('--------')
    
    print(logit_fit.predict(exog=dict(Lag1=[1.2,1.5], Lag2=[1.1,-0.8])))

    Optimization terminated successfully.
             Current function value: 0.691936
             Iterations 4
    Direction          Down  Up
    direction_predict          
    Down                 77  97
    Up                   34  44
    --------
    0.4801587301587302
    --------
    Optimization terminated successfully.
             Current function value: 0.692085
             Iterations 3
    Direction            Down   Up
    direction_pred_2var           
    Down                   35   35
    Up                     76  106
    --------
    0.5595238095238095
    --------
    0    0.479146
    1    0.496094
    dtype: float64


<a id="org4f2ae5b"></a>

### Linear Discriminant Analysis

Now we will perform LDA on `Smarket` data.

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from statsmodels import datasets
    import pandas as pd
    import numpy as np
    
    smarket = datasets.get_rdataset('Smarket', 'ISLR').data
    smarket_train = smarket.loc[smarket['Year'] < 2005]
    smarket_test = smarket.loc[smarket['Year'] == 2005].copy()
    
    lda_model = LDA()
    lda_fit = lda_model.fit(smarket_train[['Lag1', 'Lag2']],
    			smarket_train['Direction'])
    
    print(lda_fit.priors_)          # Prior probabilities of groups
    print('--------')
    print(lda_fit.means_)           # Group means
    print('--------')
    print(lda_fit.scalings_)        # Coefficients of linear discriminants
    print('--------')
    lda_predict_2005 = lda_fit.predict(smarket_test[['Lag1', 'Lag2']])
    print(pd.crosstab(lda_predict_2005, smarket_test['Direction']))
    print('--------')
    print(np.mean(lda_predict_2005 == smarket_test['Direction']))
    print('--------')
    lda_predict_prob2005 = lda_fit.predict_proba(smarket_test[['Lag1', 'Lag2']])
    print(np.sum(lda_predict_prob2005[:,0] >= 0.5))
    print(np.sum(lda_predict_prob2005[:,0] < 0.5))

    [0.49198397 0.50801603]
    --------
    [[ 0.04279022  0.03389409]
     [-0.03954635 -0.03132544]]
    --------
    [[-0.64201904]
     [-0.51352928]]
    --------
    Direction  Down   Up
    row_0               
    Down         35   35
    Up           76  106
    --------
    0.5595238095238095
    --------
    70
    182


<a id="org79fb297"></a>

### Quadratic Discriminant Analysis

We will now fit a QDA model to the `Smarket` data.

    from statsmodels import datasets
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    import pandas as pd
    import numpy as np
    
    smarket = datasets.get_rdataset('Smarket', 'ISLR').data
    smarket_train = smarket.loc[smarket['Year'] < 2005]
    smarket_test = smarket.loc[smarket['Year'] == 2005].copy()
    
    qdf = QDA()
    qdf.fit(smarket_train[['Lag1', 'Lag2']], smarket_train['Direction'])
    
    print(qdf.priors_)              # Prior probabilities of groups
    print('--------')
    print(qdf.means_)               # Group means
    print('--------')
    predict_direction2005 = qdf.predict(smarket_test[['Lag1', 'Lag2']])
    print(pd.crosstab(predict_direction2005, smarket_test['Direction']))
    print('--------')
    print(np.mean(predict_direction2005 == smarket_test['Direction']))

    [0.49198397 0.50801603]
    --------
    [[ 0.04279022  0.03389409]
     [-0.03954635 -0.03132544]]
    --------
    Direction  Down   Up
    row_0               
    Down         30   20
    Up           81  121
    --------
    0.5992063492063492


<a id="org657807b"></a>

### K-Nearest Neightbors

We will now perform KNN, also on the `Smarket` data.

    from statsmodels import datasets
    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    import numpy as np
    
    smarket = datasets.get_rdataset('Smarket', 'ISLR').data
    smarket_train = smarket.loc[smarket['Year'] < 2005]
    smarket_test = smarket.loc[smarket['Year'] == 2005].copy()
    
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(smarket_train[['Lag1', 'Lag2']], smarket_train['Direction'])
    smarket_test['predict_dir_knn1'] = knn1.predict(smarket_test[['Lag1', 'Lag2']])
    print(pd.crosstab(smarket_test['predict_dir_knn1'], smarket_test['Direction']))
    print('--------')
    print(np.mean(smarket_test['predict_dir_knn1'] == smarket_test['Direction']))
    print('--------')
    
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(smarket_train[['Lag1', 'Lag2']], smarket_train['Direction'])
    smarket_test['predict_dir_knn3'] = knn3.predict(smarket_test[['Lag1', 'Lag2']])
    print(pd.crosstab(smarket_test['predict_dir_knn3'], smarket_test['Direction']))
    print('--------')
    print(np.mean(smarket_test['predict_dir_knn3'] == smarket_test['Direction']))

    Direction         Down  Up
    predict_dir_knn1          
    Down                43  58
    Up                  68  83
    --------
    0.5
    --------
    Direction         Down  Up
    predict_dir_knn3          
    Down                48  55
    Up                  63  86
    --------
    0.5317460317460317


<a id="org104c999"></a>

### An Application to Caravan Insurance Data

Finally, we will apply the KNN approach to the `Caravan` data set in the `ISLR`
library. 

    from statsmodels import datasets
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    import numpy as np
    
    caravan = datasets.get_rdataset('Caravan', 'ISLR').data
    print(caravan['Purchase'].value_counts())
    print('--------')
    
    caravan_scale = caravan.iloc[:,:-1]
    caravan_scale = (caravan_scale - caravan_scale.mean()) / caravan_scale.std()
    
    caravan_test = caravan_scale.iloc[:1000]
    purchase_test = caravan.iloc[:1000]['Purchase']
    
    caravan_train = caravan_scale.iloc[1000:]
    purchase_train = caravan.iloc[1000:]['Purchase']
    
    # Fit KNN with 1, 3, and 5 neighbors
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(caravan_train, purchase_train)
    purchase_predict_knn1 = knn1.predict(caravan_test)
    
    print(np.mean(purchase_test != purchase_predict_knn1))
    print('--------')
    print(np.mean(purchase_test == 'Yes'))
    print('--------')
    print(pd.crosstab(purchase_predict_knn1, purchase_test))
    print('--------')
    
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(caravan_train, purchase_train)
    purchase_predict_knn3 = knn3.predict(caravan_test)
    
    print(np.mean(purchase_test != purchase_predict_knn3))
    print('--------')
    print(np.mean(purchase_test == 'Yes'))
    print('--------')
    print(pd.crosstab(purchase_predict_knn3, purchase_test))
    print('--------')
    
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(caravan_train, purchase_train)
    purchase_predict_knn5 = knn5.predict(caravan_test)
    
    print(np.mean(purchase_test != purchase_predict_knn5))
    print('--------')
    print(np.mean(purchase_test == 'Yes'))
    print('--------')
    print(pd.crosstab(purchase_predict_knn5, purchase_test))
    print('--------')
    
    # Now fit logistic regression
    logit_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    logit_model.fit(caravan_train, purchase_train)
    purchase_predict_logit = logit_model.predict(caravan_test)
    print(pd.crosstab(purchase_predict_logit, purchase_test))
    print('--------')
    
    purchase_predict_prob_logit = logit_model.predict_proba(caravan_test)
    purchase_predict_logit_prob25 = np.vectorize(
        lambda x: 'Yes' if x > 0.25 else 'No')(purchase_predict_prob_logit[:,1])
    print(pd.crosstab(purchase_predict_logit_prob25, purchase_test))

    No     5474
    Yes     348
    Name: Purchase, dtype: int64
    --------
    0.118
    --------
    0.059
    --------
    Purchase   No  Yes
    row_0             
    No        873   50
    Yes        68    9
    --------
    0.074
    --------
    0.059
    --------
    Purchase   No  Yes
    row_0             
    No        921   54
    Yes        20    5
    --------
    0.066
    --------
    0.059
    --------
    Purchase   No  Yes
    row_0             
    No        930   55
    Yes        11    4
    --------
    Purchase   No  Yes
    row_0             
    No        934   59
    Yes         7    0
    --------
    Purchase   No  Yes
    row_0             
    No        917   48
    Yes        24   11


<a id="orga6304fb"></a>

# Resampling Methods


<a id="org0c4d8a0"></a>

## Cross-Validation

Figure [143](#org76887c5) displays the *validation set approach*, a simple
stategy to estimate the test error associated with fitting a particular
statistical learning method on a set of observations.

![img](figures/fig5_1.png "A schematic display of the validation set approach.  A set of \(n\) observations are randomly split into a training set (shown in blue, containing observations 7, 22, and 13, among others) and a validation set (shown in red, and containing observation 91, among others).  The statistical learning method is fit on the training set, and its performance is evaluated on the validation set.")

In figure [145](#orgf47bf0a), the left-hand panel shows validation sample MSE as a
function of polynomial order for which a regression model was fit on training
sample.  The two samples are obtained by randomly splitting `Auto` data set into
two data sets of 196 observations each.  The right-hand panel shows the results
of repeating this exercise 10 times, each time with a different random split of
the observations into training and validation sets.  The model with a quadratic
term has a lower MSE compared to the model with only a linear term.  There is
not much benefit from adding cubic or higher order polynomial terms in the
regression model.

![img](figures/fig5_2.png "The validation set approach was used in the `Auto` data set in order to estimate the test error that results from predicting `mpg` using polynomial functions of `horsepower`.  Left: Validation error estimates for a single split into training and validation data sets.  Right: The validatioin method was repeated ten times, each time using a different random split of the observations into a training set and a validation set.  This illustrates the variability of of the estimated test MSE that results from this approach.")

Figure [147](#org415fa0b) displas the Leave One Out Cross Validation (LOOCV) approach.

![img](figures/fig5_3.png "A schematic display of LOOCV.  A set of \(n\) data points is repeatedly split into a training set (shown in blue) containing all but one observation, and a validation set that contains only that observation (shown in red).  The test error is then estimated by averaging the n resulting MSE's.  The first training set contains all but observation 1, the second training set contains all but observation 2, and so on.")

The left-hand panel of figure [149](#orge65c85a) shows test set MSE as a function of
polynomial degree when LOOCV is used on the `Auto` data set.  We fit linear
regression models to predict `mpg` using polynomial functions of `horsepower`.
The right-hand panel of figure [149](#orge65c85a) shows nine different 10-fold
CV estimates for the `Auto` data set, each resulting from a different random
split of the observations into ten folds.

![img](figures/fig5_4.png "Cross-validation was used in the `Auto` data set in order to estimate the test error that results from predicting `mpg` using polynomial functions of `horsepower`.  Left: The LOOCV error curve.  Right: 10-fold CV was run nin separate times, each with a different random split of the data into ten parts.  The figure shows the nine slightly different CV error curves.")

Figure [151](#orge4c1480) illustrates the *k*-fold CV approach.

![img](figures/fig5_5.png "A schematic display of 5-fold CV.  A set of \(n\) observations is randomly split into five non-overlapping groups.  Each of these fifths acts as a validation set (shown in red), and the remainder as a training set (shown in blue).  The test error is estimated by averaging the five resulting MSE estimates.")

In figure [153](#orged4ae5f), we plot the cross-validation estimates and true
test error rates that result from fitting least squares polynomials to the
simulated data sets illustrated in figures [21](#orgf001635),
[23](#orga1b7587), and [25](#org662b33c) of chapter [2](#org4192780).  In all
three plots, the two cross validation errors are very similar.  

![img](figures/fig5_6.png "True and estimated test MSE for the simulated data sets in Figures [21](#orgf001635) (left), [23](#orga1b7587) (center), and [25](#org662b33c) (right).  The true test MSE is shown in blue, the LOOCV estimate is shown in black dashed line, and the 10-fold CV estimate is shown in red dotted line.")

Figure [155](#orgbb375ee) shows Bayesian decision boundary (blue dashed line)
and logistic regression decision boundary (black line) for 1- to 4-degree
polynomials on \(X_1\) and \(X_2\). 

![img](figures/fig5_7.png "Logistic regression fits on the two-dimensional classification data displayed in figure [29](#org76384cb).  The Bayes decision boundary is represented using a blue dashed line.  Estimated decision boundaries from linear, quadratic, cubic, and quartic (degrees 1-4) logistic regressions are displayed in black.")

The left-hand panel of figure [158](#org4f7da17) displays in black 10-fold CV
error rates that result from fitting ten logistic regression models to the data,
using polynomial functions of the predictors up to tenth order.  The true test
errors are shown in red, and the training errors are shown in blue.  The
training error tends to decrease as the flexibility of the fit increases.  The
test error is higher than training error.  The 10-fold CV
error rate is a close approximation to the test error rate.  

The right-hand panel of figure [158](#org4f7da17) displays the same three curves
using the KNN approach for classification, as a function of the value of *K*
(the number of neighbors used in the KNN classifier).  Again, the training error
rate declines as the method becomes more flexible, and so we see that the
training error rate cannot be used to select the optimal value of *K*.  

![img](figures/fig5_8.png "Test error (red), training error(blue), and 10-fold CV error (black) on the two-dimensional classification data displayed in [155](#orgbb375ee).  Left: Logistic regression using polynomial functions of the predictors.  The order of the polynomials used is displayed on the x-axis.  Right: The KNN classifier with different values of K, the number of neighbors used in the KNN classifier.")


<a id="org8d2c9b6"></a>

## The Bootstrap


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> The middle panel is from a different data set.
