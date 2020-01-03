import numpy as np
import pandas as pd

def SMA(data, N=5):
    if 'SMA' not in data.columns:
        data = data.join(pd.DataFrame({'SMA':np.zeros(len(data))}))
    data.loc[0, 'SMA'] = data.iloc['Close'][0]
    for i in range(1,N):
        chunk = data['Close'][0:i]
        data.loc[i, 'SMA'] = np.sum(chunk) / len(chunk)
    for i in range(N,len(data)):
        chunk = data['Close'][i-N:i]
        data.loc[i, 'SMA'] = np.sum(chunk) / N
    return data

def EMA(data, P=0.5):
    if 'EMA' not in data.columns:
        data = data.join(pd.DataFrame({'EMA':np.zeros(len(data))}))
    emal = lambda v, prev: P * v + prev * (1-P)
    data.loc[0, 'EMA'] = data['Close'][0]
    for i in range(1,len(data)):
        data.loc[i, 'EMA'] = emal(data['Close'][i], data['EMA'][i-1])
    return data

