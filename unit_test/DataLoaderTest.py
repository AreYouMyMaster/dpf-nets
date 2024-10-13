import sys
import os
import numpy as np
import unittest
import torch
from lib.networks.utils import save_model

from lib.datasets.pointflowDataLoader import (
    PointflowDataLoader,
    ModelName,
    get_all_path,
)


class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        all_path = get_all_path(
            "/rawdata3/ShapeNetCore.v2.PC15k", ModelName.airplane, True
        )
        all_path.sort()
        print(all_path[0])
        self.assertEqual(all_path[0], "/rawdata3/ShapeNetCore.v2.PC15k/02691156/train/10155655850468db78d106ce0a280f87.npy")

    def test_save_data(self):
        a = torch.tensor([0,1,2,3,4])
        path = './data/models/DPFNets/a'
        save_model(a, path)
        self.assertTrue(os.path.exists(path))
        if(os.path.exists(path)):
            os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
