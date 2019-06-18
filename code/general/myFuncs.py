# Functions used in various parts of code
import pandas as pd


def dfToList(my_df, hline_after=[0]):
    '''Assumes my_df is a pandas dataframe
    Outputs a list of lists suitable for printing as org-mode table'''

    # Each list within list will be a row in the table
    # Header needs an extra column for index names
    res_headers = list(my_df.columns)
    res_headers.insert(0, my_df.index.name)

    res_list = my_df.values.tolist()
    for i in range(len(res_list)):
        res_list[i].insert(0, my_df.index.tolist()[i])

    res_list.insert(0, res_headers)
    hline_count = 0
    for loc in hline_after:
        hline_count += 1
        res_list.insert(hline_count + loc, None)

    # res_list.insert(1, None)
    return res_list
