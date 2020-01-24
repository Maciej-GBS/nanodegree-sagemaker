import sagemaker_containers
import torch
import torch.nn

class LSTMRegressor(torch.nn.Module):
    def __init__(self, conv_kernel_size, hidden_dim):
        super().__init__()
        # TODO: Create torch nn layers
        # Each channel is a different column of our data
        # So input of 10 bars history has a shape of (N,C,L) = (N,n_columns,10)
        self.conv = torch.nn.Conv1d(input_channels, conv_channels, conv_kernel_size)
        self.lstm = torch.nn.LSTM(conv_channels * (input_dim - conv_kernel_size - 1), hidden_dim)
        self.dense = None
        self.sigmoid = None
    
    def forward(self, x):
        # TODO: create a correct forward
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sigmoid(out.squeeze())
    
class LSTMBatchRegressor(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        # TODO: Create torch nn layers
        self.norm = torch.nn.BatchNorm1d(num_features)
        self.lstm = torch.nn.LSTM(num_features, hidden_dim)
    
    def forward(self, x):
        # TODO: create a correct forward
        pass

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
    model = LSTMRegressor(model_info['my_dim'])

    # Restore model
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("> Model loading: Finished")
    return model
