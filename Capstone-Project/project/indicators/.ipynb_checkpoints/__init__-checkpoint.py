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
def Gap(data, timedelta):
    """
    Time gap between candles.
    Requires: numpy.timedelta64
    Column: Gap
    """
    gap_loc = data.columns.get_loc('Gap')
    data = data.astype({'Gap':'timedelta64[ns]'})
    
    for i in range(1,len(data)):
        data.iloc[i, gap_loc] = data.iloc[i].name - data.iloc[i-1].name
    data.iloc[:, gap_loc] = data.iloc[:, gap_loc] / timedelta
    return data

@column_add_wrapper
def SMA(data, N=5):
    """
    Simple moving average.
    Column: SMA
    """
    sma_loc = data.columns.get_loc('SMA')
    
    data.iloc[0, sma_loc] = data.iloc[0]['Close']

    # Until expected length is reached divide by length of chunk
    for i in range(1,N):
        chunk = data['Close'][0:i]
        data.iloc[i, sma_loc] = np.sum(chunk) / len(chunk)

    for i in range(N,len(data)):
        chunk = data['Close'][i-N:i]
        data.iloc[i, sma_loc] = np.sum(chunk) / N
    return data

@column_add_wrapper
def EMA(data, P=0.5):
    """
    Exponential moving average.
    Column: EMA
    """
    ema_loc = data.columns.get_loc('EMA')
    
    emal = lambda v, prev: P * v + prev * (1-P)
    data.iloc[0, ema_loc] = data.iloc[0]['Close']
    for i in range(1,len(data)):
        data.iloc[i, ema_loc] = emal(data.iloc[i]['Close'], data.iloc[i-1]['EMA'])
    return data

@column_add_wrapper
def Momentum(data):
    """
    Calculates momentum value: (Close - PrevClose).
    Column: Momentum
    """
    mom_loc = data.columns.get_loc('Momentum')
    
    for i in range(1,len(data)):
        data.iloc[i, mom_loc] = data.iloc[i]['Close'] - data.iloc[i-1]['Close']
    return data

@column_add_wrapper
def RSI(data, period=14):
    """
    Calculates RSI oscillator. Requires Momentum.
    Column: RSI
    """
    rsi_loc = data.columns.get_loc('RSI')
    
    def process_chunk(chunk, length):
        # Average Gain
        AG = (chunk > 0).sum() / length
        # Average Loss
        AL = (chunk < 0).sum() / length
        RS = AG / (AL + 1e-12)
        data.iloc[i, rsi_loc] = 100.0 - 100.0 / (1 + RS)
        
    for i in range(0,period):
        change = np.array(data.iloc[0:i]['Momentum'])
        process_chunk(change, i+1)
    
    for i in range(period,len(data)):
        change = np.array(data.iloc[i-period:i]['Momentum'])
        process_chunk(change, period)
    return data
