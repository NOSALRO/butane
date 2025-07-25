import unittest
import torch
import butane

class TestMathOps(unittest.TestCase):

    def test_sum_around(self):
        x = torch.randn(2, 3, 4, 5)
        reduced_x = butane.sum_around(x, 1)
        self.assertEqual(x.shape[1], len(reduced_x))

        reduced_x = butane.sum_around(x, (1, 2))
        correct = True
        for rs, xs in zip(reduced_x.shape, [x.shape[1], x.shape[2]]):
            if rs != xs:
                correct = False
        self.assertTrue(correct)

    def test_mean_around(self):
        x = torch.randn(2, 3, 4, 5)
        reduced_x = butane.mean_around(x, 1)
        self.assertEqual(x.shape[1], len(reduced_x))

        reduced_x = butane.mean_around(x, (1, 2))
        correct = True
        for rs, xs in zip(reduced_x.shape, [x.shape[1], x.shape[2]]):
            if rs != xs:
                correct = False
        self.assertTrue(correct)

    def test_max_around(self):
        x = torch.randn(2, 3, 4, 5)
        max_v = butane.apply_around_dim(torch.max, x, 1)
        print(max_v.size())

if __name__ == '__main__':
    unittest.main()
