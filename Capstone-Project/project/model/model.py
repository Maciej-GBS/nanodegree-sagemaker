#import sagemaker_containers
import torch
import torch.nn

class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_size, input_channels, c_filters, c_kernel_size, lstm_layers, lstm_hidden, output_size):
        super().__init__()
        # Each channel is a different column of our data
        # So input of 11 bars history has a shape of (N,C,L) = (len,batch,features) = (N,n_columns,11)
        c_out_channels = input_channels * c_filters
        self.conv = torch.nn.Conv1d(in_channels=input_channels, out_channels=c_out_channels, kernel_size=c_kernel_size)
        c_out = input_size - c_kernel_size + 1
        self.lstm = torch.nn.LSTM(input_size=c_out, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=0.2)
        lstm_flat_out = lstm_hidden * c_out_channels
        self.dense = torch.nn.Linear(in_features=lstm_flat_out, out_features=output_size)
    
    def forward(self, x):
        print("===={0}=====\n{1}\n==========".format("x",x.shape))
        conv_out = self.conv(x.transpose(-2,-1))
        print("===={0}=====\n{1}\n==========".format("conv_out",conv_out.shape))
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = lstm_out.flatten(start_dim=1)
        print("===={0}=====\n{1}\n==========".format("lstm_out",lstm_out.shape))
        #dense_in = torch.cat((lstm_out.transpose(-2,-1), x[:,-1,3].reshape(lstm_out.shape[0], lstm_out.shape[2], 1)), dim=2).transpose(-2,-1)
        dense_in = lstm_out
        print("===={0}=====\n{1}\n==========".format("dense_in",dense_in.shape))
        dense_out = self.dense(dense_in).squeeze()
        return dense_out
    
class LSTMBatchRegressor(torch.nn.Module):
    # TODO: This is a copy of CONV Regressor !!!
    def __init__(self, input_size, input_channels, c_filters, c_kernel_size, lstm_layers, lstm_hidden, output_size):
        super().__init__()
        # Each channel is a different column of our data
        # So input of 11 bars history has a shape of (N,C,L) = (len,batch,features) = (N,n_columns,11)
        c_out_channels = input_channels * c_filters
        self.conv = torch.nn.Conv1d(in_channels=input_channels, out_channels=c_out_channels, kernel_size=c_kernel_size)
        c_out = input_size - c_kernel_size + 1
        self.lstm = torch.nn.LSTM(input_size=c_out, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=0.2)
        lstm_flat_out = lstm_hidden * c_out_channels
        self.dense = torch.nn.Linear(in_features=lstm_flat_out, out_features=output_size)
    
    def forward(self, x):
        print("===={0}=====\n{1}\n==========".format("x",x.shape))
        conv_out = self.conv(x.transpose(-2,-1))
        print("===={0}=====\n{1}\n==========".format("conv_out",conv_out.shape))
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = lstm_out.flatten(start_dim=1)
        print("===={0}=====\n{1}\n==========".format("lstm_out",lstm_out.shape))
        #dense_in = torch.cat((lstm_out.transpose(-2,-1), x[:,-1,3].reshape(lstm_out.shape[0], lstm_out.shape[2], 1)), dim=2).transpose(-2,-1)
        dense_in = lstm_out
        print("===={0}=====\n{1}\n==========".format("dense_in",dense_in.shape))
        dense_out = self.dense(dense_in).squeeze()
        return dense_out

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("> Loading model...")

    # Read model init arguments from model_info
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)
    print("> model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: correct parameters
    model = LSTMRegressor(model_info['my_dim'])

    # Restore model
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("> Model loading: Finished")
    return model
