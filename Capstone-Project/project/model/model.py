import sagemaker_containers
import torch
import torch.nn

class LSTMRegressor(torch.nn.Module):
    def __init__(self, my_dim):
        super().__init__()
        
        self.lstm = None
        self.dense = None
        self.sigmoid = None
    
    def forward(self, x):
        # WARNING
        # THIS is example code
        # TODO: wrimoidte correct function
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sigmoid(out.squeeze())

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
