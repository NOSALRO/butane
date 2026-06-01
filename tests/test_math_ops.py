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

    def test_odeint(self):
        func = lambda t, x: (x[0], x[1])
        x = torch.randn(4, 5)
        y = torch.zeros_like(x)
        print(butane.math.odeint(func, (x, y), torch.linspace(0, 1, 10), 'heun2', return_func_outputs=True))

if __name__ == '__main__':
    unittest.main()
