import sys
import os
import numpy as np
sys.path.insert(0, "../lib/datasets")
import unittest
from pointflowDataLoader import PointflowDataLoader, ModelName


class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        dataloader = PointflowDataLoader('/rawdata3/ShapeNetCore.v2.PC15k', ModelName.airplane, True)
        for path in dataloader.get_all_path():
            print(path)
    
if __name__ == '__main__':
    unittest.main(verbosity=2)