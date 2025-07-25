import unittest
import torch
import butane

class TestPCGrad(unittest.TestCase):

    def test_pcgrad(self):
        model = butane.nn.MLPBlock(
            input_dims=10,
            output_dims=3,
            hidden_dims=[16, 16, 32, 64],
            activation_function=[torch.nn.ReLU()]
        )
        optim = torch.optim.Adam(model.parameters())
        x = torch.randn(10).requires_grad_(True)
        pred = model(x)
        loss1 = (pred - torch.randn(3)).pow(2).mean()
        loss2 = -(pred - torch.randn(3)).pow(2).sum()
        butane.optim.pcgrad([loss1, loss2], optim, model)

if __name__ == '__main__':
    unittest.main()
