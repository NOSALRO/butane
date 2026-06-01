import unittest
import torch
import butane

class TestScaler(unittest.TestCase):

    def test_standard_scaler(self):
        x = torch.rand(100, 3, 20, 20)
        scaler = butane.data.StandardScaler()
        scaler.fit(x, 1)
        x = scaler(x)
        x = scaler(x, inverse=True)
        print(f"Mean: {x.mean()}, STD: {x.std()}")


    def test_minmax_scaler(self):
        x = torch.randn((10, 3, 20, 20))
        scaler = butane.data.MinMaxScaler()
        scaler.fit(x, 1)
        x = scaler(x)
        print(f"Min: {x.min()}, Max: {x.max()}")

if __name__ == '__main__':
    unittest.main()
