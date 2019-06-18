# Functions to select all subsets, forward stepwise, and backward stepwise

import statsmodels.formula.api as smf
from itertools import combinations
import numpy as np
from operator import attrgetter


def allSubsets(y_var, all_x, train_df):
    '''Assumes y_var is the dependent variable, all_x is a list of all
    explanatory variables.  train_df is dataframe whose columns include
    y_var and all elements of all_x.  Returns a dictionary with a complete list
    of models when one, two, ..., all variables are selected.'''
    all_models = {}

    for p in range(1, len(all_x) + 1):
        all_models[p] = {}
        all_models[p]['model'] = []
        all_models[p]['x_vars'] = []
        x_combinations = combinations(all_x, p)
        for select_var in x_combinations:
            my_formula = y_var + ' ~ ' + ' + '.join(select_var)
            lm_model = smf.ols(my_formula, data=train_df)
            lm_fit = lm_model.fit()
            all_models[p]['model'].append(lm_fit)
            all_models[p]['x_vars'].append(select_var)
    return all_models


def bestSubset(all_models, metric, metric_max=True, func_or_attr='attr'):
    '''Assumes all_models is a dictionary output by allSubsets, and metric
    is a metric output by statsmodels.  For every key value in all_models,
    use metric to find best model and its variables.  Return best model
    and its variables.'''
    num_vars = []
    best_model = []
    best_model_vars = []
    best_model_metric = []

    if func_or_attr == 'attr':  # metric is output by statsmodels fit
        getMetric = attrgetter(metric)
    elif func_or_attr == 'func':  # metric is calculated by function
        getMetric = metric

    for p in all_models.keys():
        metric_vals = []
        for model in all_models[p]['model']:
            metric_vals.append(getMetric(model))

        if metric_max:          # higher value of metric is more desirable
            best_ind = np.argmax(metric_vals)
        else:
            best_ind = np.argmin(metric_vals)

        num_vars.append(p)
        best_model.append(all_models[p]['model'][best_ind])
        best_model_vars.append(all_models[p]['x_vars'][best_ind])
        best_model_metric.append(metric_vals[best_ind])

    return (num_vars, best_model, best_model_vars, best_model_metric)


def forwardStepSelect(y_var, all_x, train_df, metric, metric_max=True):
    '''Assumes y_var is the dependent variable, all_x is a list of all
    explanatory variables.  train_df is a dataframe whose columns include
    y_var and all elements of all_x.  metric (an output of statsmodels) is used
    to select best model at every step.  If metric_max is True, then 
    higher value indicates better model.  If metrix_max is False, then
    lower value indicates better model.'''
    remaining_x = all_x.copy()
    best_models = {}
    getMetric = attrgetter(metric)
    my_formula = y_var + ' ~ '

    for p in range(1, len(all_x) + 1):
        best_models[p] = {}
        best_models[p]['all_x'] = []
        if p > 1:
            for x in best_models[p-1]['all_x']:
                best_models[p]['all_x'].append(x)

        if metric_max:
            best_metric = 0
        else:
            best_metric = np.inf

        for next_x in remaining_x:
            new_formula = my_formula + ' + ' + next_x
            lm_model = smf.ols(new_formula, data=train_df)
            lm_fit = lm_model.fit()
            model_metric = getMetric(lm_fit)
            model_update = (metric_max and model_metric > best_metric) or \
                ((not metric_max) and model_metric < best_metric)
            if model_update:
                best_models[p]['model'] = lm_fit
                best_models[p]['next_x'] = next_x
                best_models[p]['metric'] = model_metric
                best_metric = model_metric
                selected_x = next_x

        remaining_x.remove(selected_x)
        best_models[p]['all_x'].append(selected_x)
        my_formula = my_formula + ' + ' + selected_x
    return best_models


def backwardStepSelect(y_var, all_x, train_df, metric, metric_max=True):
    '''Start with all_x variables.  Use backward step selection method to 
    drop least useful variable one by one.  metric (an output of statsmodels)
    is used to select best model at every step.  y_var is dependent variable.
    train_df is a dataframe whose columns include y_var and all x_var.
    Returns a dictionary of best models, one for each count.'''
    remaining_x = all_x.copy()
    best_models = {}
    getMetric = attrgetter(metric)

    p = len(all_x)
    my_formula = y_var + ' ~ ' + ' + '.join(remaining_x)
    lm_model = smf.ols(my_formula, data=train_df)
    lm_fit = lm_model.fit()
    model_metric = getMetric(lm_fit)

    best_models[p] = {}
    best_models[p]['model'] = lm_fit
    best_models[p]['all_x'] = remaining_x.copy()
    best_models[p]['metric'] = model_metric

    # pdb.set_trace()
    for p in range(len(all_x) - 1, 0, -1):
        best_models[p] = {}
        if metric_max:
            best_metric = 0
        else:
            best_metric = np.inf

        for drop_x in remaining_x:
            temp_remaining_x = remaining_x.copy()
            temp_remaining_x.remove(drop_x)
            my_formula = y_var + ' ~ ' + ' + '.join(temp_remaining_x)
            lm_model = smf.ols(my_formula, data=train_df)
            lm_fit = lm_model.fit()
            model_metric = getMetric(lm_fit)
            model_update = (metric_max and model_metric > best_metric) or \
                ((not metric_max) and model_metric < best_metric)
            if model_update:
                best_models[p]['model'] = lm_fit
                best_models[p]['all_x'] = temp_remaining_x
                best_models[p]['metric'] = model_metric
                best_metric = model_metric
                selected_drop_x = drop_x
        remaining_x.remove(selected_drop_x)
    return best_models


def C_p(model):
    '''Calculate Cp coefficient for model fit using statsmodels'''
    n = model.df_model + model.df_resid + 1
    RSS = model.ssr
    d = model.df_model
    sigma_hat_sq = model.mse_resid
    cp = (RSS + 2.0 * d * sigma_hat_sq) / n
    return cp
