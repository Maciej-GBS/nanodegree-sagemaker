import os.path
import torch
import torch.nn
import torch.utils.data

def denormalize(y, close):
    """ Denormalize output of Regressor. Requires Close (predicted) input channel. """
    mean = torch.mean(close, dim=1).reshape(y.shape[0],1).repeat(1,y.shape[1])
    std = torch.std(close, dim=1).reshape(y.shape[0],1).repeat(1,y.shape[1])
    return std*y + mean
    
class LSTMRegressor(torch.nn.Module):
    """
    Parameters:
    input_size,
    input_channels (columns),
    c_filters (convolution filters),
    c_kernel_size (1-D filter size),
    lstm_layers,
    lstm_hidden (lstm layer output count),
    dropout (probability to zero a value),
    output_size (prediction horizon in a single forward pass)
    """
    def __init__(self, input_size, input_channels, c_filters, c_kernel_size, lstm_layers, lstm_hidden, dropout, output_size):
        super().__init__()
        self.size = input_size
        self.channels = input_channels
        # Each channel is a different column of our data
        # So input of 11 bars history has a shape of (N,C,L) = (len,batch,features) = (N,n_columns,11)
        c_out_channels = input_channels * c_filters
        c_out = input_size - c_kernel_size + 1
        lstm_flat_out = lstm_hidden * c_out_channels
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.norm = torch.nn.BatchNorm1d(input_channels)
        #self.c_filter = torch.normal(mean=torch.tensor([0.5,-1.0,0.5]),std=torch.tensor([0.1,0.3,0.1]))
        #self.c_filter = (self.c_filter * torch.ones(c_out_channels,input_channels,1)).requires_grad_()
        #self.c_bias = torch.zeros(c_out_channels).requires_grad_() # Parameters for functional convolution
        self.conv = torch.nn.Conv1d(in_channels=input_channels, out_channels=c_out_channels, kernel_size=c_kernel_size)
        self.lstm = torch.nn.LSTM(input_size=c_out, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=lstm_dropout)
        self.drop = torch.nn.Dropout(p=dropout)
        self.dense = torch.nn.Linear(in_features=lstm_flat_out, out_features=output_size)
    
    def forward(self, x):
        norm_out = self.norm(x.transpose(-2,-1))
        #self.c_filter = self.c_filter.type(x.dtype).to(x.device) # Functional implementation of convolution
        #self.c_bias = self.c_bias.type(x.dtype).to(x.device)
        #conv_out = torch.nn.functional.conv1d(x.transpose(-2,-1), self.c_filter, bias=self.c_bias)
        conv_out = self.conv(norm_out)
        lstm_out, _ = self.lstm(conv_out)
        dense_in = self.drop(lstm_out.flatten(start_dim=1))
        dense_out = self.dense(dense_in)
        
        # It is possible to normalize the output here,
        # but this requires assumption of the Close channel
        # position in the x tensor.
        # Instead we return raw output and allow the script
        # above to use denormalize function later.
        return dense_out
    
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("> Loading model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read model init arguments from model_info
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)
    print("> model_info: {}".format(model_info))

    # Rebuild model from info
    model = LSTMRegressor(input_size=model_info['input_size'],
                          input_channels=len(model_info['input_channels']),
                          c_filters=model_info['c_filters'],
                          c_kernel_size=model_info['c_kernel_size'],
                          lstm_layers=model_info['lstm_layers'],
                          lstm_hidden=model_info['lstm_hidden'],
                          dropout=model_info['dropout'],
                          output_size=model_info['output_size'])

    # Restore model
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.double().to(device).eval()

    print("> Model loading: Finished")
    return model

class SlidingWindowDataset(torch.utils.data.Dataset):
    """
    Iterable Dataset from numpy array data containing history of market.
    Slides a window over the dataset, ensures that output has the shape of
    (window_size, data_columns)
    """
    def __init__(self, data, window_size):
        super().__init__()
        self.timeseries = data
        self.window = window_size
        self.length = len(self.timeseries)
    
    def __len__(self):
        return self.length - self.window + 1
        
    def __getitem__(self, index):
        if type(index) is slice:
            start = 0
            stop = self.length - self.window
            step = 1
            if index.start is not None:
                start = index.start
            if index.stop is not None:
                stop = index.stop
            if index.step is not None:
                step = index.step
            return np.array([self.timeseries[i:i+self.window] for i in range(start,stop,step)])
        else:
            last = index + self.window
            if index < 0 or last > self.length:
                raise IndexError
            return self.timeseries[index:last]
