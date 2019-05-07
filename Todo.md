# Anomaly detection framework TODO list

- [ ] Implement the first algorithm selected: DeepAnt
    - [ ] Implement the data ingestion mechanism
        - [X] define the data file in the configuration file
        - [X] ensure we create a DataLoader() object
        - [X] define the percentage of data in the training set in the config file
        - [ ] implement code to load HDF5 files in addition to CSV files
    - [ ] Implement the time series predictor
    - [ ] Implement the anomaly detector
    - [ ] Implement automatic thresholding methods
- [ ] Test the algorithm with data
- [ ] Refactor the code
- [ ] Start implementing MSCRED on the same basis as DeepAnt

