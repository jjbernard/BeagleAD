# This file will load the configuration for the selected algorithm and create a model
# using the Predictor class corresponding to the algorithm and using a DataLoader object

import json

with open('config.json') as config_data_file:
    config = json.load(config_data_file)

# Store sequence size and prediction window size
w = config['general']['w']
p_w = config['general']['w']

# Identify algorithms to use
algs = []
for key, value in config['methods'].items():
    algs.append(value)

# Store parameters for algorithms
parameters = dict()

