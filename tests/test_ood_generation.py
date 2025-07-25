import unittest
import torch
import butane

class TestODIN(unittest.TestCase):

    def test_odin(self):
        model = butane.nn.MLPBlock(
            input_dims=10,
            output_dims=3,
            hidden_dims=[16, 16],
            activation_function=[torch.nn.ReLU()]
        )
        x = torch.randn(10).requires_grad_(True)
        pred = model(x)
        loss = (pred - torch.randn(3)).pow(2).mean()
        loss.backward()
        ood_sample = butane.nn.functional.odin(x, epsilon=20, normalize=False)
        self.assertIsNotNone(x)

if __name__ == '__main__':
    unittest.main()
