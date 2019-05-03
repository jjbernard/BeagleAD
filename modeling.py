# This file will load the configuration for the selected algorithm and 
# create a model using the XXPredictor class corresponding to the 
# selected algorithm with a DataLoader object

import json
from algs import DAPredictor

with open('config.json') as config_data_file:
    config = json.load(config_data_file)

# Store sequence size and prediction window size
w = config['general']['w']
p_w = config['general']['p_w']
max_epochs = config['general']['epochs']

# Identify algorithms to use
algs = config['methods']['algs']

# Store parameters for algorithms
parameters = dict()

for alg in algs:
    parameters[alg] = config['algos'][alg]

# Load DataSet and DataLoader first
# identify the number of time series from the DataSet / DataLoader
# Dummy values at this stage

nb_ts = 1

# Method to return the model to create. There is probably a more elegant way
# to do this...
def modelSelector(name, params):
    if name == "deepant":
        return DAPredictor(nb_ts, params[name]['nb_filters'], 
                            params[name]['kernel_size_conv'],
                            params[name]['kernel_size_pool'], w, p_w)

# Loop over the methods selected to train them

for alg in algs:
    model = modelSelector(alg, parameters)
    for epoch in range(max_epochs):
        pass