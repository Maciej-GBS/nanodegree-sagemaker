"""
Utils containg space consuming, simple functions.

Imports: pandas, numpy.
"""
import pandas as pd
import numpy as np

def load_file_split(path):
    """ Merges split .csv files into single DataFrame """
    return pd.read_csv(path + 'columns-example.txt')
