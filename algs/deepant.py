# This file contains the algorithm for time series prediction as described in the DeepAnt paper.
# DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series
# by MOHSIN MUNIR, SHOAIB AHMED SIDDIQUI,ANDREAS DENGEL AND SHERAZ AHMED
# DOI: 10.1109/ACCESS.2018.2886457

import pytorch.nn as nn
import pytorch.nn.functional as F

class Predictor(nn.Module):
    """ Defines the neural network used for Prediction. 
        It is composed of the following components:
            - a convolutional layer with a ReLU activation
            - a maxpool layer
            - another convolutional layer with a ReLU activation
            - another maxpool layer
            - a fully connected layer to output the predicted value
        
        Each convolutional layer is composed of 32 filters. 
        We use a stride of 1 and no padding for the convolutional layers
    """
    def __init__(self):
        super(Predictor, self).__init__()
        self.conv1 = nn.Conv1d()
        self.maxpool1 = nn.MaxPool1d() 
        self.conv2 = nn.Conv1d()
        self.maxpool2 = nn.MaxPool1d()
        self.fc = nn.Linear()

    def forward(self, input):
        out = self.maxpool1(F.relu(self.conv1(input)))
        out2 = self.maxpool2(F.relu(self.conv2(out)))
       
        result = nn.Linear(out2)

        return result
