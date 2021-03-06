# Anomaly detection framework TODO list

- [ ] Rework the structure of the code:
    - [X] Create a fit() function to train the model
    - [X] Create a save() function to save the model
    - [X] Create a function to select the model, the optimizer, the loss 
    - [X] Create a function to validate the model
    - [ ] Create a class that encapsulate the fit() and save() functions that holds all the necessary data and information to train the model. See fast.ai (part 2) for this. This could be a Learner() class
    - [ ] Review fast.ai course (part 2) to create a databunch class specific for time series. 
    - [ ] Review and adjust the class that creates the model
    - [ ] Create a generic model class that can be derived to create subclasses for all algorithms
- [X] Continue tests in notebooks:
    - [X] Instead of using nb of channels as the number of variables in a time series, try keeping number of channels = 1 and add a dimension to the input data (through `torch.unsqueeze()`). This should allow us to keep the dimensions as they are and not permute them as it is currently done in the code. 

