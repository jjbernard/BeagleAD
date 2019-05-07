# This file contains the code used to load the time series data
# We assume data is contained in CSV files with a timestamp index

import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Data should be stored in the Data/ directory as a CSV file
# Shape of the CSV file should be N x M+1 where:
# N is the number of timestamps
# M is the number of features (M+1 because we end up with a timestamp column)
# Timestamp is considered the first column of the dataset

def createTSDataLoader(filename='data.csv', train_size, bs, w, p_w):
    """Create two dataloaders for the training and validation datasets."""

    dirpath = Path('Data')
    path = dirpath / filename
    data = pd.read_csv(path)

    data = data.iloc[:,1:]

    # We consider both training and validation data is in the same dataset
    N = len(data)

    # X will be a list of sequences of size w
    # y will be a list of sequences of size p_w, 
    # immediately following the corresponding X
    
    X = []
    y = []

    # Total sequence to go over is N + 1 - w - p_w
    seq = N + 1 - w - p_w
    for i in range(seq):
        X_temp, y_temp = data.iloc[i:i+w], data.iloc[i+w:i+w+p_w]
        X.append(X_temp)
        y.append(y_temp)

    idx = int(len(X) * train_size)

    # Needs to define x and y from data
    train_ds = TensorDataset(X[:idx], y[:idx])
    valid_ds = TensorDataset(X[idx:], y[idx:])

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=False)
    valid_ld = DataLoader(valid_ds, batch_size=bs, shuffle=False)

    return train_ld, valid_ld
