import json
#import sagemaker_containers # Module not found?
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import yfinance

from model import *

def input_fn(serialized_input_data, content_type):
    """ Input should be text symbol to predict on (e.g. EURUSD) """
    if content_type == 'text/plain':
        print("> Received input symbol")
        data = serialized_input_data.decode('utf-8')
        # TODO: fetch data from yfinance
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print("> Responding with forecast")
    return str(prediction_output)

def predict_fn(input_data, model):
    print('> Inferring forecast for provided symbol')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = torch.from_numpy(input_data)
    data = data.to(device)

    # Model into evaluation mode
    model.eval()

    with torch.no_grad():
        result = model(data)
        result = np.array(result)

    return result
