import unittest
import torch
import butane

class TestUtils(unittest.TestCase):

    def test_determnistic_sampling(self):
        foo_data = torch.ones(100, 20, 30)
        foo_data[:, :10] *= 2
        foo_data[:, 10:] *= 3
        batching = butane.batching(foo_data, 10, drop_last=False, dim=1)
        for b in batching: 
            print(b)

if __name__ == '__main__':
    unittest.main()
