import math
import torch
import numpy as np

def make_spiral(n_samples: int = 10_000, max_theta: float = 5*np.pi, is_3d: bool = False) -> torch.Tensor:
    theta = np.linspace(0., max_theta, n_samples)
    r = np.linspace(0, 1, n_samples)
    spiral = np.array([r*np.cos(theta), r*np.sin(theta)]).T
    if is_3d:
        spiral = np.concatenate((spiral, np.linspace(0, 1, n_samples)[:,None]), axis=-1)
    spiral += 0.01 * np.random.randn(*spiral.shape)
    return torch.tensor(spiral).float()

def make_eight_normal(
    n_samples: int = 10_000,
    scale: float = 1.,
    var: float = 1.,
    n_dims: int = 2,
) -> torch.Tensor:
    dist = torch.distributions.MultivariateNormal(torch.zeros(n_dims), math.sqrt(var) * torch.eye(n_dims))
    z = np.linspace(0, 1, 4)
    centers = [
        (0, 1, z[-1]),
        (1.0 / math.sqrt(2), 1.0 / math.sqrt(2), z[-2]),
        (1, 0, z[-3]),
        (1.0 / math.sqrt(2), -1.0 / math.sqrt(2), z[-4]),
        (0, -1, z[0]),
        (-1.0 / math.sqrt(2), -1.0 / math.sqrt(2), z[1]),
        (-1, 0, z[2]),
        (-1.0 / math.sqrt(2), 1.0 / math.sqrt(2), z[3]),
    ]
    centers = torch.tensor(centers)[:, :n_dims] * scale
    noise = dist.sample((n_samples, ))
    multi = torch.multinomial(torch.ones(len(centers)), n_samples, replacement=True)
    data = []
    for i in range(n_samples):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data.float()

def make_moons(n_samples: int = 10_000, noise_coef: float = 0.05) -> torch.Tensor:
    n_samples_1 = n_samples // 2
    n_samples_2 = n_samples - n_samples_1
    x1 = torch.cos(torch.linspace(0, torch.pi, n_samples_1))
    y1 = torch.sin(torch.linspace(0, torch.pi, n_samples_1))
    x2 = 1 - torch.cos(torch.linspace(0, torch.pi, n_samples_2))
    y2 = 1 - torch.sin(torch.linspace(0, torch.pi, n_samples_2)) - 0.5

    c1 = torch.cat([x1[...,None], y1[...,None]], dim=-1)
    c2 = torch.cat([x2[...,None], y2[...,None]], dim=-1)
    labels = torch.cat([torch.zeros((c1.size(0),)), torch.ones((c2.size(0),))], dim=0)
    moons = torch.cat([c1, c2], dim=0)
    shuffled_indices = torch.randperm(moons.size(0))
    moons = moons[shuffled_indices]
    labels = labels[shuffled_indices]
    moons = moons + noise_coef * torch.randn_like(moons)
    return moons, labels
