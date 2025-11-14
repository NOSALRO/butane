import unittest
import torch
import butane
import matplotlib.pyplot as plt

class TestPCGrad(unittest.TestCase):

    def test_ddpm(self):
        torch.manual_seed(0)
        image = torch.zeros((3, 28, 28)).to('cuda')
        image[:, 5:24, 12:16] = 1.
        num_timesteps = 1000
        diffusion = butane.nn.ImprovedDDPM(num_timesteps, scheduler='cosine').to('cuda')
        t = diffusion.sample_timestep(image.size(0))
        x_t, eps = diffusion(image, t)
        sampled = diffusion.sample(
            model=lambda x: x,
            x_T = torch.randn_like(image),
        )
        print(sampled.size())


if __name__ == '__main__':
    unittest.main()
