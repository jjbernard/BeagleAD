# This file contains the algorithm for time series prediction as described in the DeepAnt paper.
# DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series
# by MOHSIN MUNIR, SHOAIB AHMED SIDDIQUI,ANDREAS DENGEL AND SHERAZ AHMED
# DOI: 10.1109/ACCESS.2018.2886457

import torch.nn as nn
import torch.nn.functional as F

class DAPredictor(nn.Module):
    """ Defines the neural network used for Prediction. 
        It is composed of the following components:
            - a convolutional layer with a ReLU activation
            - a maxpool layer
            - another convolutional layer with a ReLU activation
            - another maxpool layer
            - a fully connected layer to output the predicted value
        
        Each convolutional layer is composed of 32 filters. 
        We use a stride of 1 and no padding for the convolutional layers
        Args:
            - nb_ts: number of time series
            - nb_filters: number of filters for convolution layers
            - kernel_size_conv: size of the filters for the convolution layers
            - kernel_size_pool: size of the filters for the max pool layers
            - w: size of the sequence
            - p_w: number of steps in the future to predict
    """
    def __init__(self, nb_ts, nb_filters, kernel_size_conv, kernel_size_pool, w, p_w):
        super(DAPredictor, self).__init__()
        # Input for nn.Conv1d() is nb input channels, nb output channels, kernel size
        # The number of channel is going to be 1
        self.nb_ts = nb_ts
        self.nb_filters = nb_filters
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_pool = kernel_size_pool
        self.w = w
        self.p_w = p_w
        self.conv1 = nn.Conv1d(1, nb_filters, kernel_size_conv, padding=1, bias=True)
        self.maxpool = nn.MaxPool1d(kernel_size_pool)
        self.conv2 = nn.Conv1d(nb_filters,nb_filters, kernel_size_conv, padding=1, bias=True)
        # Need a way to calculate the size of the series after all the previous operations
        self.fc = nn.Linear(w - 2 * (kernel_size_pool - 1),p_w)

    def forward(self, input):
        out = self.maxpool(F.relu(self.conv1(input)))
        out2 = self.maxpool(F.relu(self.conv2(out)))
    
        # Need to call out2.view(nb_ts,-1) to flatten before calling nn.Linear()
        # with nb_ts which is the number of time series (i.e. 1 for univariate 
        # and more than one for multivariate)
        out2 = out2.view(self.nb_ts, -1)

        result = F.relu(self.fc(out2))
 
        return result
