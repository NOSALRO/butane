import unittest
import torch
import butane

class TestTransforms(unittest.TestCase):

    def test_transforms(self):
        x = torch.randn((10, 20, 30))
        t = butane.data.Transforms(
            lambda x: x.flatten(1),
        )
        t = t + (lambda x: x.flatten())

        self.assertTrue(t(x).size(0) == (10 * 20 * 30))

if __name__ == '__main__':
    unittest.main()
