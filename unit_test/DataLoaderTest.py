import sys
import os
import numpy as np
import unittest

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
        self.assertEqual(all_path[0], "10155655850468db78d106ce0a280f87.npy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
