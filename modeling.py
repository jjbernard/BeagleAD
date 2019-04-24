# This file will load the configuration for the selected algorithm and create a model
# using the Predictor class corresponding to the algorithm and using a DataLoader object

import json

with open('config.json') as config_data_file:
    config = json.load(config_data_file)

print(config)
