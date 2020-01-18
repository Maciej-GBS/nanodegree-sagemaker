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
