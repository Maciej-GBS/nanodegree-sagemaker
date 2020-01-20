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
            empty_col = pd.DataFrame({fn.__name__: np.zeros(len(data))}, index=data.index)
            data = pd.concat( [ data,  empty_col], axis=1)
        return fn(data, *args, **kwargs)
    return wrapped

@column_add_wrapper
def SMA(data, N=5):
    """
    Simple moving average.
    Column: SMA
    """
    data.iloc[0, data.columns.get_loc('SMA')] = data.iloc[0]['Close']

    # Until expected length is reached divide by length of chunk
    for i in range(1,N):
        chunk = data['Close'][0:i]
        data.iloc[i, data.columns.get_loc('SMA')] = np.sum(chunk) / len(chunk)

    for i in range(N,len(data)):
        chunk = data['Close'][i-N:i]
        data.iloc[i, data.columns.get_loc('SMA')] = np.sum(chunk) / N
    return data

@column_add_wrapper
def EMA(data, P=0.5):
    """
    Exponential moving average.
    Column: EMA
    """
    emal = lambda v, prev: P * v + prev * (1-P)
    data.iloc[0, data.columns.get_loc('EMA')] = data.iloc[0]['Close']
    for i in range(1,len(data)):
        data.iloc[i, data.columns.get_loc('EMA')] = emal(data.iloc[i]['Close'], data.iloc[i-1]['EMA'])
    return data

@column_add_wrapper
def Momentum(data):
    """
    Calculates momentum value: (Close - PrevClose).
    Column: Momentum
    """
    for i in range(1,len(data)):
        data.iloc[i, data.columns.get_loc('Momentum')] = data.iloc[i]['Close'] - data.iloc[i-1]['Close']
    return data

@column_add_wrapper
def RSI(data, period=14):
    """
    Calculates RSI oscillator. Requires Momentum.
    Column: RSI
    """
    for i in range(0,period):
        chunk = np.array(data.iloc[0:i]['Momentum'])
        # Average Gain
        AG = chunk[chunk > 0].sum() / period
        # Average Loss
        AL = chunk[chunk < 0].sum() / period
        data.iloc[i, data.columns.get_loc('RSI')] = 100.0 - 100.0 / (1 + AG / AL)
    
    for i in range(period,len(data)):
        chunk = np.array(data.iloc[i-period:i]['Momentum'])
        # Average Gain
        AG = chunk[chunk > 0].sum() / period
        # Average Loss
        AL = chunk[chunk < 0].sum() / period
        data.iloc[i, data.columns.get_loc('RSI')] = 100.0 - 100.0 / (1 + AG / AL)
    return data
