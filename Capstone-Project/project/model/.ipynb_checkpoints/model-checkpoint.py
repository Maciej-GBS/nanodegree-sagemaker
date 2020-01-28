import torch
import torch.nn

class LSTMRegressor(torch.nn.Module):
    """
    This model uses convolution to normalize input.
    Finally predicts change to most recent input Close value, which is added in the final output.
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
        # Each channel is a different column of our data
        # So input of 11 bars history has a shape of (N,C,L) = (len,batch,features) = (N,n_columns,11)
        c_out_channels = input_channels * c_filters
        self.conv = torch.nn.Conv1d(in_channels=input_channels, out_channels=c_out_channels, kernel_size=c_kernel_size)
        c_out = input_size - c_kernel_size + 1
        self.lstm = torch.nn.LSTM(input_size=c_out, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=dropout)
        lstm_flat_out = lstm_hidden * c_out_channels
        self.drop = torch.nn.Dropout(p=dropout)
        self.dense = torch.nn.Linear(in_features=lstm_flat_out, out_features=output_size)
    
    def forward(self, x):
        conv_out = self.conv(x.transpose(-2,-1))
        lstm_out, _ = self.lstm(conv_out)
        dense_in = self.drop(lstm_out.flatten(start_dim=1))
        dense_out = self.dense(dense_in)
        # Add latest Close values to our output
        close = x[:,-1,3].reshape(x.shape[0])
        return dense_out + (close * torch.ones(dense_out.shape[1],1)).t()
    
class LSTMBatchRegressor(torch.nn.Module):
    """
    LSTM regression with batch normalization on input.
    Output is denormalized with average and standard deviation from Close channel.
    Parameters:
    input_size,
    input_channels (columns),
    lstm_layers,
    lstm_hidden (lstm layer output count),
    dropout (probability to zero a value),
    output_size (prediction horizon in a single forward pass)
    """
    def __init__(self, input_size, input_channels, lstm_layers, lstm_hidden, dropout, output_size):
        super().__init__()
        # Each channel is a different column of our data
        # So input of 11 bars history has a shape of (N,C,L) = (len,batch,features) = (N,n_columns,11)
        self.norm = torch.nn.BatchNorm1d(num_features=input_channels)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=dropout)
        lstm_flat_out = lstm_hidden * input_channels
        self.drop = torch.nn.Dropout(p=dropout)
        self.dense = torch.nn.Linear(in_features=lstm_flat_out, out_features=output_size)
    
    def forward(self, x):
        norm_out = self.norm(x.transpose(-2,-1))
        lstm_out, _ = self.lstm(norm_out)
        dense_in = self.drop(lstm_out.flatten(start_dim=1))
        dense_out = self.dense(dense_in)
        # Denormalize with std and mean from Close channel
        std = (torch.std(x[:,:,3], dim=1) * torch.ones(dense_out.shape[1],1)).t()
        mean = (torch.mean(x[:,:,3], dim=1) * torch.ones(dense_out.shape[1],1)).t()
        return std * dense_out + mean

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
    if model_info['type']==1:
        model = LSTMBatchRegressor(input_size=model_info['input_size'],
                                   input_channels=model_info['input_channels'],
                                   lstm_layers=model_info['lstm_layers'],
                                   lstm_hidden=model_info['lstm_hidden'],
                                   dropout=model_info['dropout'],
                                   output_size=model_info['output_size'])
    else:
        model = LSTMRegressor(input_size=model_info['input_size'],
                              input_channels=model_info['input_channels'],
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

    model.to(device).eval()

    print("> Model loading: Finished")
    return model
