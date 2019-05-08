# Tests directory

Will need to store all the test files the different capabilities are working properly

## test_loader.py

This test file will check:
- that the data is properly loaded into 2 different dataloaders with a split of 80%/20% between them.
- that when calling a dataloader, a batch size is of the correct size and the unitary item from the batch size correspond to the specified size (i.e. number of time series, data points used for prediction and size of target data to predict)