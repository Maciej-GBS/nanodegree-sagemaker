#import sagemaker_containers # Module not found?
import json
import numpy as np
import pandas as pd
import indicators
import torch
import torch.utils.data
import yfinance as yf

from model import *

def convert(data, horizon, meta, ema=0.03, sma=70, rsi=14):
    # Get specific columns Open High Low Close
    data = data.loc[:,['Open','High','Low','Close']]
    # Apply indicators
    data = indicators.Gap(data, np.timedelta64(1,'D'))
    data = indicators.EMA(data, P=ema)
    data = indicators.SMA(data, N=sma)
    data = indicators.Momentum(data)
    data = indicators.RSI(data, period=rsi)
    # Return only meta channels
    return torch.from_numpy(data.iloc[-horizon:, meta].values)

def input_fn(serialized_input_data, content_type):
    """ Input should be text symbol to predict on (e.g. EURUSD) """
    if content_type != 'text/plain':
        raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
    print("> Received symbol information")
    
    symbol = serialized_input_data.decode('utf-8')
    interval = '1d'
    period = '6mo'
    sym = yf.Ticker(symbol)
    data = sym.history(period=period, interval=interval)
    return data

def output_fn(prediction_output, accept):
    print("> Responding with forecast")
    d = {
        'x':prediction_output[1].tolist(),
        'open':prediction_output[0][:,0].tolist(),
        'high':prediction_output[0][:,1].tolist(),
        'low':prediction_output[0][:,2].tolist(),
        'close':prediction_output[0][:,3].tolist(),
        }
    return json.dumps(d)

def predict_fn(input_data, model):
    print('> Inferring forecast for provided symbol')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    market = convert(input_data, model.size, model.channels)
    # Expected size is (1, size, channels)
    data = market.unsqueeze(dim=0).to(device)

    # Model into evaluation mode
    model.eval()
    with torch.no_grad():
        result = model(data)
        result = np.array(denormalize(result, data[:,:,3]))
        
    x = np.array(input_data.index)
    x = np.append(x, x[-1] + np.timedelta64(1,'D'))
    market = market.numpy()
    result = np.append(market[:,:4], [[result]*4], axis=0)
    return np.array([result, x])
