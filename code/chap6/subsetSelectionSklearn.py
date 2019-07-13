# Use sklearn to find best model with exhaustive enumeration
# sklearn is faster than statsmodels, but less flexible.
# This is because model metrics (e.g., R-squared) have to be calculated.
# bestSubset function only implements RSS (residual sum of squares) to select
# the best subset.  It is straightforward to add other metrics.

from itertools import combinations
from sklearn.linear_model import LinearRegression
import numpy as np


def getVarLookup(var_num, var_cat, var_cat_dummies):
    '''Assumes var_num is list of numeric variable names.
    var_cat is list of categorical variable names.
    var_cat_dummies is list of names of categorical variables converted to
    dummy variables.  Returns dictionary of variable mapping.'''

    var_lookup = {}
    for name in var_num:
        var_lookup[name] = [name]

    for name in var_cat:
        var_lookup[name] = []
        for name_dummy in var_cat_dummies:
            if name in name_dummy:
                if (len(name) + 3) > len(name_dummy):
                    var_lookup[name].append(name_dummy)

    return var_lookup


def bestSubset(y, x_vars_numeric, x_vars_cat, x_var_dummies,
               train_df, subset_size=1):
    '''Assumes y is the dependent variable vector.
    x_vars_numeric is a list of numeric independent variables.
    x_vars_cat is a list of categorical indepenent variables.
    x_var_dummies is the list of dummy variables obtained from x_vars_cat.
    train_df is dataframe whose columns include all numeric and dummy 
    variables.  
    subset_size is the number of variables for which best model is to be
    selected.  Returns best model for which RSS is minimized.'''

    var_lookup = getVarLookup(x_vars_numeric, x_vars_cat,
                              x_var_dummies)
    all_x = x_vars_numeric.copy()
    all_x.extend(x_vars_cat)
    x_combinations = combinations(all_x, subset_size)

    best_metric = np.inf
    for select_var in x_combinations:
        x_columns = []
        for x_name in select_var:
            x_columns.extend(var_lookup[x_name])

        X = train_df[x_columns]
        reg_model = LinearRegression()
        reg_model.fit(X, y)
        y_fit = reg_model.predict(X)
        model_metric = np.sum((y_fit - y) ** 2)
        if model_metric < best_metric:
            best_model = reg_model
            best_x_vars = select_var
            best_var_dummies = x_columns
            best_metric = model_metric
    return {'model': best_model, 'x_var_names': best_x_vars,
            'var_numeric_dummies': best_var_dummies,
            'metric': best_metric, 'metric_name': 'RSS'}


# Calculate test error metrics by making adjustment to training error
# These metrics are implemented in statsmodels, but not in sklearn
def testStats(X, y, model):
    '''X is training dataframe whose columns include explantaory variables
    y is training set dependent variable
    model is a dictionary output from function bestSubset
    Returns a dictionary with Cp, AIC, BIC, and adjusted R-squared'''

    x_columns = list(model['var_numeric_dummies'])
    y_predict = model['model'].predict(X[x_columns])
    RSS = np.sum((y_predict - y) ** 2)  # residual sum of squares
    TSS = np.sum((y - np.mean(y)) ** 2)  # total sum of squares
    n_obs = X.shape[0]
    p_vars = model['model'].rank_
    adj_rsq = 1 - (n_obs - 1) * RSS / ((n_obs - p_vars - 1) * TSS)
    llf = (-n_obs / 2) * (np.log(2 * np.pi) + np.log(RSS / n_obs) + 1)

    # Estimate error variance when all variables are included
    lm_model = LinearRegression()
    lm_model.fit(X, y)
    sigma_hat_sq = np.sum((lm_model.predict(X) - y) ** 2) / \
        (n_obs - lm_model.rank_ - 1)

    C_p = (RSS + 2 * p_vars * sigma_hat_sq) / n_obs
    AIC = -2 * llf + 2 * (p_vars + 1)
    BIC = -2 * llf + np.log(n_obs) * (p_vars + 1)
    return {'adj_rsq': adj_rsq, 'C_p': C_p, 'AIC': AIC, 'BIC': BIC}
