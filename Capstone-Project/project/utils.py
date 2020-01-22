"""
Utils containg space consuming, simple functions.

Imports: os, pandas, numpy.
"""
import pandas as pd
import numpy as np
import os

def load_file_split(path):
    """ Merges split .csv files into single DataFrame """
    names = pd.read_csv(path + 'columns-example.txt').columns
    csv_files = [os.path.join(path, f) for f in os.listdir(path) if f[-4:] == '.csv']
    parts = [pd.read_csv(csv, names=names, header=None) for csv in sorted(csv_files)]
    return pd.concat(parts).reset_index(drop=True)

def change_timeframe(data, stack_size=60):
    """
    Creates new dataframe with specified stacking (new timeframe).
    Ignores indexing.
    """
    framed_df = pd.DataFrame(columns=data.columns)
    l_data = len(data)
    rows_ls = []
    for i in range(0, l_data, stack_size):
        frame = data.loc[i:] if (i+stack_size >= l_data) else data.loc[i:i+stack_size]
        row = {k:frame.iloc[-1][k] for k in data.columns}
        row.update({'Open':frame.iloc[0]['Open'], 'High':max(frame.loc[:,'High']), 'Low':min(frame.loc[:,'Low'])})
        rows_ls.append(row)
    return framed_df.append(rows_ls, ignore_index=True)
