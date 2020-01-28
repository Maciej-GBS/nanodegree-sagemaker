import argparse
import os
import json
#import sagemaker_containers # Module not found?
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from model import *

class SlidingWindowDataset(torch.utils.data.IterableDataset):
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
        
    def __next__(self):
        first = self._start + self._n
        last = self._n + self.window + self._start
        if last > self._end:
            raise StopIteration
        self._n += 1
        return self.timeseries[first:last]
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # single-process data loading, return the full iterator
            self._start = 0
            self._end = self.length
        else: # split if inside a worker process
            per_worker = int(math.ceil(self.length / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self._start = worker_id * per_worker
            self._end = min(start + per_worker, self.length)
        # window looks forward so _end has to shrinked
        self._end -= self.window
        self._n = 0
        return self

def _get_train_data_loader(batch_size, sliding_window, pair, training_dir):
    train_data = pd.read_csv(os.path.join(training_dir, "train_{}.csv".format(pair)), header=None, names=None)
    train_ds = SlidingWindowDataset(train_data.values, sliding_window)
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def train(model, outputs, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. Parameters:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (cuda or cpu).
    """
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in train_loader:         
            # Slice dataset for input and output
            batch_X = batch[:len(batch)-outputs].to(device)
            batch_Y = batch[-outputs:].to(device)
            
            y = model(batch_X)
            # Use only Close price column
            loss = loss_fn(y, batch_Y[:,3])
            loss.backward()

            # Update weights and reset gradients
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.data.item()
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=12, metavar='S',
                        help='random seed (default: 12)')
    # Model Parameters
    parser.add_argument('--type', type=int, default=0, metavar='N',
                        help='regressor type: 1-batch, 0-convolution (default: 0)')
    parser.add_argument('--input-size', type=int, default=5, metavar='N',
                        help='input window size (default: 5)')
    parser.add_argument('--input-channels', type=int, default=9, metavar='N',
                        help='expected dataset columns (default: 9)')
    parser.add_argument('--lstm-layers', type=int, default=2, metavar='N',
                        help='lstm layers used in the model (default: 2)')
    parser.add_argument('--lstm-hidden', type=int, default=1, metavar='N',
                        help='lstm layer output - horizon (default: 1)')
    parser.add_argument('--dropout', type=int, default=0.2, metavar='P',
                        help='probability to drop/zero input node (default: 0.2)')
    parser.add_argument('--output-size', type=int, default=1, metavar='N',
                        help='horizon, predictions forward (default: 1)')
    parser.add_argument('--c-filters', type=int, default=2, metavar='N',
                        help='convolution filters (default: 2)')
    parser.add_argument('--c-kernel-size', type=int, default=3, metavar='N',
                        help='convolution filter kernel size (default: 3)')
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    window_size = args.input_size + args.output_size
    train_loader = _get_train_data_loader(args.batch_size, window_size, args.pair, args.data_dir)
    
    if args.type==1:
        model = LSTMBatchRegressor(input_size=args.input_size,
                                   input_channels=args.input_channels,
                                   lstm_layers=args.lstm_layers,
                                   lstm_hidden=args.lstm_hidden,
                                   dropout=args.dropout,
                                   output_size=args.output_size)
    else:
        model = LSTMRegressor(input_size=args.input_size,
                              input_channels=args.input_channels,
                              c_filters=args.c_filters,
                              c_kernel_size=args.c_kernel_size,
                              lstm_layers=args.lstm_layers,
                              lstm_hidden=args.lstm_hidden,
                              dropout=args.dropout,
                              output_size=args.output_size)
    model = model.to(device)

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss(reduction='sum')
    train(model, args.output_size, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the model parameters to model_info
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'type': args.type,
            'input_size': args.input_size,
            'input_channels': args.input_channels,
            'lstm_layers': args.lstm_layers,
            'lstm_hidden': args.lstm_hidden,
            'dropout': args.dropout,
            'output_size': args.output_size,
            'c_filters': args.c_filters,
            'c_kernel_size': args.c_kernel_size
        }
        torch.save(model_info, f)

    # Save model
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
