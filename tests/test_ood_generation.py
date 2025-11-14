from functools import partial
import unittest
import torch
import butane

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)

    def forward(self, x, t):
        h = x + t
        return self.fc1(h)

class TestFGSM(unittest.TestCase):

    def test_fgm(self):
        model = butane.nn.MLPBlock(
            input_dims=10,
            output_dims=3,
            hidden_dims=[16, 16],
            activation_function=[torch.nn.ReLU()]
        )
        x = torch.randn(1, 10)
        ood_sample = butane.nn.functional.fgm(x, J=lambda x: (model(x) - torch.randn(3)).pow(2).mean(), epsilon=0.04)
        self.assertIsNotNone(x)

    def test_pgm(self):
        x = torch.randn(1, 10).requires_grad_(True)
        t = torch.randn(1, 10)
        x_gt = torch.ones(20)
        model = Model()
        pred = model(x, t)
        butane.nn.functional.pgm(
            x=x,
            J=partial(lambda x, y: torch.mean((model(x, t)-y)**2), y=x_gt),
            epsilon=0.3,
        )

if __name__ == '__main__':
    unittest.main()
