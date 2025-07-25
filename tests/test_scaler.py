import unittest
import torch
import butane

class TestScaler(unittest.TestCase):

    def test_standard_scaler(self):
        x = torch.arange(10).unsqueeze(0).repeat(10, 1).unsqueeze(-1).float()
        scaler = butane.data.StandardScaler()
        scaler.fit(x, 1)
        x = scaler(x)
        scaler.reverse(x[:, -1], feature_idx = 9)
        print(f"Mean: {x.mean()}, STD: {x.std()}")


    def test_minmax_scaler(self):
        x = torch.randn((10, 3, 20, 20))
        scaler = butane.data.MinMaxScaler()
        scaler.fit(x, 2)
        x = scaler(x)
        print(f"Min: {x.min()}, Max: {x.max()}")

if __name__ == '__main__':
    unittest.main()
