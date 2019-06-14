# This file will load the configuration for the selected algorithm and 
# create a model using the XXPredictor class corresponding to the 
# selected algorithm with a DataLoader object

import json
import torch
from algs.deepant import DAPredictor
from dataload import createTSDataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


with open('config.json') as config_data_file:
    config = json.load(config_data_file)

# Store sequence size and prediction window size
w = config['general']['w']
p_w = config['general']['p_w']
max_epochs = config['general']['epochs']
lr = config['general']['learning_rate']
bs = config['general']['batch_size']
train_size = config['general']['train_size']
MODELPATH = Path(config['general']['models'])

# Identify algorithms to use
algs = config['methods']['algs']

# Store parameters for algorithms
parameters = dict()

for alg in algs:
    parameters[alg] = config['algos'][alg]

# Load DataSet and DataLoader first
# identify the number of time series from the DataSet / DataLoader
# Dummy values at this stage

train_dl, valid_dl, nb_ts = createTSDataLoader(train_size, bs, w, 
                                                p_w, filename='data.csv')

# Method to return the model to create. There is probably a more elegant way
# to do this...
def modelSelector(name, params):
    if name == "deepant":
        return DAPredictor(nb_ts, params[name]['nb_filters'], 
                            params[name]['kernel_size_conv'],
                            params[name]['kernel_size_pool'], w, p_w)

if __name__ == "__main__":

    # Loop over the methods selected to train the algorithm
    for alg in algs:
        model = modelSelector(alg, parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
        for epoch in range(max_epochs):
            
            for i, data in enumerate(train_dl):
                inputs = data[0]
                labels = data[1]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        filename = MODELPATH / alg

        torch.save(model, filename)

    # Evaluate accuracy with validation dataset

    for alg in algs:
        pass


    



                