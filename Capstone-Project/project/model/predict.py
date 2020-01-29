import json
#import sagemaker_containers # Module not found?
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import yfinance as yf

from model import *

def convert(data):
    # TODO: write me
    return data

def input_fn(serialized_input_data, content_type):
    """ Input should be text symbol to predict on (e.g. EURUSD) """
    if content_type != 'application/json':
        raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
    print("> Received input symbol")
    sym_info = serialized_input_data.decode('utf-8')
    interval = sym_info['interval'] if sym_info['interval'] in ['1m','2m','5m','15m','30m','60m','90m','1h','1d'] else '1h'
    period = '1mo' if ('d' in interval) else '5d'
    sym = yf.Ticker(sym_info['symbol'])
    data = sym.history(period=period, interval=interval)
    return data

def output_fn(prediction_output, accept):
    print("> Responding with forecast")
    return str(prediction_output)

def predict_fn(input_data, model):
    print('> Inferring forecast for provided symbol')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: make this work
    data = convert(input_data).to(device)

    # Model into evaluation mode
    model.eval()
    with torch.no_grad():
        result = model(data)
        result = np.array(result)
    return result
