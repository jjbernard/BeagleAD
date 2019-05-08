# Test file to ensure created DataLoaders (both train and valid)
# are working properly

import unittest
from dataload import createTSDataLoader

class DataLoadTests(unittest.TestCase):

    def setUp(self):
        self.train_size = 0.8
        self.bs = 64
        self.w = 20
        self.p_w = 2
        self.data_file = 'data.csv'
        self.train_dl, self.valid_dl = createTSDataLoader(filename='data.csv', 
                                                            train_size=self.train_size,
                                                            bs = self.bs,
                                                            w = self.w,
                                                            p_w = self.pw)

    def tearDown(self):
        pass

    def test_load():
        pass
    

if __name__ == "__main__":
    unittest.main()