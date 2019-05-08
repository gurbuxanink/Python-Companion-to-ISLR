
# Table of Contents

1.  [Introduction](#org95d78fa)
2.  [Statistical Learning](#orgb1124a2)



<a id="org95d78fa"></a>

# Introduction

Figure [2](#org61becc9) shows graphs of Wage versus three variables. 

![img](figures/fig1_1.png "`Wage` data, which contains income survey information for males from the central Atlantic region of the United States.  Left: `wage` as a function of `age`.  On average, `wage` increases with `age` until about 60 years of age, at which point it begins to decline.  Center: `wage` as a function of `year`.  There is a slow but steady increase of approximately $10,000 in the average `wage` between 2003 and 2009.  Right: Boxplots displaying `wage` as a function of `education`, with 1 indicating the lowest level (no highschool diploma) and 5 the highest level (an advanced graduate degree).  On average, `wage` increases with the level of `education`.")

Figure [4](#orgd5d31e7) shows boxplots of previous days' percentage changes in S&P
500 grouped according to today's change `Up` or `Down`. 

![img](figures/fig1_2.png "Left: Boxplots of the previous day's percentage change in the S&P 500 index for the days for which the market increased or decreased, obtained from the `Smarket` data.  Center and Right: Same as left panel, but the percentage changes for two and three days previous are shown.")


<a id="orgb1124a2"></a>

# Statistical Learning

Figure [6](#org12f330a) shows scatter plots of `sales` versus `TV`, `radio`,
and `newspaper` advertising.  In each panel, the figure also includes an OLS
regression line.  

![img](figures/fig2_1.png "The `Advertising` data set. The plot displays `sales`, in thousands of units, as a function of `TV`, `radio`, and `newspaper` budgets, in thousands of dollars, for 200 different markets.  In each plot we show the simple least squares fit of `sales` to that variable.  In other words, each red line represents a simple model that can be used to predict `sales` using `TV`, `radio`, and `newspaper`, respectively.")

Figure [8](#org9d81f10) is a plot of `Income` versus `Years of Education` from the
Income data set.  In the left panel, the \`\`true'' function (given by blue line)
is actually my guess.  

![img](figures/fig2_2.png "The `Income` data set.  Left: The red dots are the observed values of `income` (in tens of thousands of dollars) and `years of education` for 30 individuals.  Right: The blue curve represents the true underlying relationship between `income` and `years of education`, which is generally unknown (but is known in this case because the data are simulated).  The vertical lines represent the error associated with each observation.  Note that some of the errors are positive (when an observation lies above the blue curve) and some are negative (when an observation lies below the curve).  Overall, these errors have approximately mean zero.")

Figure [10](#org2bb6e72) is a plot of `Income` versus `Years of Education` and
`Seniority` from the `Income` data set.  Since the book does not provide the
true values of `Income`, \`\`true'' values shown in the plot are actually third
order polynomial fit.  

![img](figures/fig2_3.png "The plot displays `income` as a function of `years of education` and `seniority` in the `Income` data set.  The blue surface represents the true underlying relationship between `income` and `years of education` and `seniority`, which is known since the data are simulated.  The red dots indicate the observed values of these quantities for 30 individuals.")

Figure [12](#org6f2433d) shows an example of the parametric approach applied to
the `Income` data from previous figure. 

![img](figures/fig2_4.png "A linear model fit by least squares to the `Income` data from figure [10](#org2bb6e72).  The observations are shown in red, and the blue plane indicates the least squares fit to the data.")

Figure [14](#org854415b) provides an illustration of the trade-off between
flexibility and interpretability for some of the methods covered in this book.

![img](figures/figure2_7.png "A representation of the tradeoff between flexibility and interpretability, using different statistical learning methods.  In general, as the flexibility of a method increases, its interpretability decreases.")

Figure [16](#orgcc84660) provides a simple illustration of the clustering problem.

![img](figures/fig2_8.png "A clustering data set involving three groups.  Each group is shown using a different colored symbol.  Left: The three groups are well-separated.  In this setting, a clustering approach should successfully identify the three groups.  Right: There is some overlap among the groups.  Now the clustering taks is more challenging.")

