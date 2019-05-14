
# Table of Contents

1.  [Introduction](#orgd3c0bfa)
2.  [Statistical Learning](#orgd10ae97)
    1.  [What is Statistical Learning?](#orgc0d1c8e)
    2.  [Assessing Model Accuracy](#org4019891)



<a id="orgd3c0bfa"></a>

# Introduction

Figure [2](#org30f0d84) shows graphs of Wage versus three variables. 

![img](figures/fig1_1.png "`Wage` data, which contains income survey information for males from the central Atlantic region of the United States.  Left: `wage` as a function of `age`.  On average, `wage` increases with `age` until about 60 years of age, at which point it begins to decline.  Center: `wage` as a function of `year`.  There is a slow but steady increase of approximately $10,000 in the average `wage` between 2003 and 2009.  Right: Boxplots displaying `wage` as a function of `education`, with 1 indicating the lowest level (no highschool diploma) and 5 the highest level (an advanced graduate degree).  On average, `wage` increases with the level of `education`.")

Figure [4](#orgdacd9f6) shows boxplots of previous days' percentage changes in S&P
500 grouped according to today's change `Up` or `Down`. 

![img](figures/fig1_2.png "Left: Boxplots of the previous day's percentage change in the S&P 500 index for the days for which the market increased or decreased, obtained from the `Smarket` data.  Center and Right: Same as left panel, but the percentage changes for two and three days previous are shown.")


<a id="orgd10ae97"></a>

# Statistical Learning


<a id="orgc0d1c8e"></a>

## What is Statistical Learning?

Figure [6](#orgbbc7524) shows scatter plots of `sales` versus `TV`, `radio`,
and `newspaper` advertising.  In each panel, the figure also includes an OLS
regression line.  

![img](figures/fig2_1.png "The `Advertising` data set. The plot displays `sales`, in thousands of units, as a function of `TV`, `radio`, and `newspaper` budgets, in thousands of dollars, for 200 different markets.  In each plot we show the simple least squares fit of `sales` to that variable.  In other words, each red line represents a simple model that can be used to predict `sales` using `TV`, `radio`, and `newspaper`, respectively.")

Figure [8](#org4dab69e) is a plot of `Income` versus `Years of Education` from the
Income data set.  In the left panel, the \`\`true'' function (given by blue line)
is actually my guess.  

![img](figures/fig2_2.png "The `Income` data set.  Left: The red dots are the observed values of `income` (in tens of thousands of dollars) and `years of education` for 30 individuals.  Right: The blue curve represents the true underlying relationship between `income` and `years of education`, which is generally unknown (but is known in this case because the data are simulated).  The vertical lines represent the error associated with each observation.  Note that some of the errors are positive (when an observation lies above the blue curve) and some are negative (when an observation lies below the curve).  Overall, these errors have approximately mean zero.")

Figure [10](#org3a05069) is a plot of `Income` versus `Years of Education` and
`Seniority` from the `Income` data set.  Since the book does not provide the
true values of `Income`, \`\`true'' values shown in the plot are actually third
order polynomial fit.  

![img](figures/fig2_3.png "The plot displays `income` as a function of `years of education` and `seniority` in the `Income` data set.  The blue surface represents the true underlying relationship between `income` and `years of education` and `seniority`, which is known since the data are simulated.  The red dots indicate the observed values of these quantities for 30 individuals.")

Figure [12](#org5c82468) shows an example of the parametric approach applied to
the `Income` data from previous figure. 

![img](figures/fig2_4.png "A linear model fit by least squares to the `Income` data from figure [10](#org3a05069).  The observations are shown in red, and the blue plane indicates the least squares fit to the data.")

Figure [14](#orgf660e34) provides an illustration of the trade-off between
flexibility and interpretability for some of the methods covered in this book.

![img](figures/figure2_7.png "A representation of the tradeoff between flexibility and interpretability, using different statistical learning methods.  In general, as the flexibility of a method increases, its interpretability decreases.")

Figure [16](#org9cd852a) provides a simple illustration of the clustering problem.

![img](figures/fig2_8.png "A clustering data set involving three groups.  Each group is shown using a different colored symbol.  Left: The three groups are well-separated.  In this setting, a clustering approach should successfully identify the three groups.  Right: There is some overlap among the groups.  Now the clustering taks is more challenging.")


<a id="org4019891"></a>

## Assessing Model Accuracy

Figure [20](#org0e891e4) illustrates the tradeoff between training MSE and test
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

When we repeat the simulations for figure [20](#org0e891e4), we see considerable
variation in the right panel MSE plots.  But the overall conclusion remains the
same.   

![img](figures/fig2_9.png "Left: Data simulated from \(f\), shown in black.  Three estimates of \(f\) are shown: the linear regression line (orange curve), and two smoothing spline fits (blue and green curves).  Right: Training MSE (grey curve), test MSE (red curve), and minimum possible test MSE over all methods (dashed grey line).")

Figure [22](#orgdbe35c8) provides another example in which the true \(f\) is
approximately linear. 

![img](figures/fig2_10.png "Details are as in figure [20](#org0e891e4) using a different true \(f\) that is much closer to linear.  In this setting, linear regression provides a very good fit to the data.")

Figure [24](#orga04b92e) displays an example in which \(f\) is highly
non-linear. The training and test MSE curves still exhibit the same general
patterns.

![img](figures/fig2_11.png "Details are as in figure [20](#org0e891e4), using a different \(f\) that is far from linear.  In this setting, linear regression provides a very poor fit to the data.")

Figure [26](#orgf18e6d2) displays the relationship between bias, variance, and
test MSE.  This relationship is referred to as *bias-variance trade-off*.  When
simulations are repeated, we see considerable variation in different graphs,
especially for MSE lines.  But overall shape remains the same. 

![img](figures/fig2_12.png "Squared bias (blue curve), variance (orange curve), \(Var(\epsilon)\) (dashed line), and test MSE (red curve) for the three data sets in figures [20](#org0e891e4) - [24](#orga04b92e).  The vertical dotted line indicates the flexibility level corresponding to the smallest test MSE.")

Figure [28](#org6a1a484) provides an example using a simulated data set in
two-dimensional space consisting of predictors \(X_1\) and \(X_2\).  

![img](figures/fig2_13.png "A simulated data set consisting of 200 observations in two groups, indicated in blue and orange.  The dashed line represents the Bayes decision boundary.  The orange background grid indicates the region in which a test observation will be assigned to the orange class, and blue background grid indicates the region in which a test observation will be assigned to the blue class.")

Figure [30](#orgef4de7c) displays the KNN decision boundary, using \(K=10\), when
applied to the simulated data set from figure [28](#org6a1a484).  Even though
the true distribution is not known by the KNN classifier, the KNN decision
making boundary is very close to that of the Bayes classifier.  

![img](figures/fig2_15.png "The firm line indicates the KNN decision boundary on the data from figure [28](#org6a1a484), using \(K = 10\). The Bayes decision boundary is shown as a dashed line.  The KNN and Bayes decision boundaries are very similar.")

![img](figures/fig2_16.png "A comparison of the KNN decision boundaries (solid curves) obtained using \(K=1\) and \(K=100\) on the data from figure [28](#org6a1a484).  With \(K=1\), the decision boundary is overly flexible, while with \(K=100\) it is not sufficiently flexible.  The Bayes decision boundary is shown as dashed line.")

In figure [33](#org5dc55c1) we have plotted the KNN test and training errors as
a function of \(\frac{1}{K}\).  As \(\frac{1}{K}\) increases, the method becomes
more flexible.  As in the regression setting, the training error rate
consistently declines as the flexibility increases.  However, the test error
exhibits the characteristic U-shape, declining at first (with a minimum at
approximately \(K=10\)) before increasing again when the method becomes
excessively flexible and overfits. 

![img](figures/fig2_17.png "The KNN training error rate (blue, 200 observations) and test error rate (orange, 5,000 observations) on the data from figure [28](#org6a1a484) as the level of flexibility (assessed using \(\frac{1}{K}\)) increases, or equivalently as the number of neighbors \(K\) decreases.  The black dashed line indicates the Bayes error rate.")

