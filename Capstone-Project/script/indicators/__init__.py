"""
Module with indicator applying functions.
Each function appends or recalculates desired indicator on provided dataset.
Expected dataset columns: Close, High, Low, Open.

Imports: numpy, pandas, functools.
"""
import numpy as np
import pandas as pd
from functools import wraps

def column_add_wrapper(fn):
    @wraps(fn)
    def wrapped(data, *args, **kwargs):
        if fn.__name__ not in data.columns:
            data = data.join(pd.DataFrame({fn.__name__:np.zeros(len(data))}))
        return fn(data, *args, **kwargs)
    return wrapped

@column_add_wrapper
def SMA(data, N=5):
    """
    Simple moving average.
    Column: SMA
    """
    data.loc[0, 'SMA'] = data.loc['Close'][0]

    # Until expected length is reached divide by length of chunk
    for i in range(1,N):
        chunk = data['Close'][0:i]
        data.loc[i, 'SMA'] = np.sum(chunk) / len(chunk)

    for i in range(N,len(data)):
        chunk = data['Close'][i-N:i]
        data.loc[i, 'SMA'] = np.sum(chunk) / N
    return data

@column_add_wrapper
def EMA(data, P=0.5):
    """
    Exponential moving average.
    Column: EMA
    """
    emal = lambda v, prev: P * v + prev * (1-P)
    data.loc[0, 'EMA'] = data.loc['Close'][0]
    for i in range(1,len(data)):
        data.loc[i, 'EMA'] = emal(data['Close'][i], data['EMA'][i-1])
    return data

@column_add_wrapper
def Momentum(data):
    """
    Calculates momentum value: (Close - PrevClose).
    Column: Momentum
    """
    for i in range(1,len(data)):
        data.loc[i, 'Momentum'] = data['Close'][i] - data['Close'][i-1]
    return data
