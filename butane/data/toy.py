from typing import Tuple, Callable
import math
import torch
import numpy as np

def make_spiral(
    n_samples: int = 10_000,
    max_theta: float = 5*np.pi,
    sigma: float = 0.01,
    regression: bool = False,
    seq_len: int = 5,
    is_3d: bool = False
) -> torch.Tensor:
    theta = np.linspace(0., max_theta, n_samples)
    r = np.linspace(0, 1, n_samples)
    spiral = np.array([r*np.cos(theta), r*np.sin(theta)]).T
    if is_3d:
        spiral = np.concatenate((spiral, np.linspace(0, 1, n_samples)[:,None]), axis=-1)
    spiral += sigma * np.random.randn(*spiral.shape)
    spiral_tensor = torch.tensor(spiral).float()
    if not regression:
        return spiral_tensor

    num_valid_samples = n_samples - seq_len
    data = spiral_tensor[:num_valid_samples] # (N, 2)
    targets_view = spiral_tensor[1:].unfold(0, seq_len, 1)
    targets = targets_view[:num_valid_samples].permute(0, 2, 1)
    return data, targets

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

def make_moons(n_samples: int = 10_000, noise_coef: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
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

def make_pinwheel(n_samples: int = 2000, n_classes: int = 5, noise: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    points_per_class = n_samples // n_classes
    data = []
    labels = []
    for k in range(n_classes):
        r = torch.linspace(0.1, 1.0, points_per_class)
        theta = torch.linspace(0, 2.5, points_per_class) + (k * 2 * np.pi / n_classes)
        x = (r + torch.randn(points_per_class) * noise) * torch.cos(theta)
        y = (r + torch.randn(points_per_class) * noise) * torch.sin(theta)
        data.append(torch.stack([x, y], dim=1))
        labels.append(torch.full((points_per_class,), k))
    return torch.cat(data), torch.cat(labels)

def make_ring(n_samples: int = 2000, r_min: float = 0.3, r_max: float = 0.6) -> torch.Tensor:
    theta = torch.rand(n_samples) * 2 * np.pi
    r = torch.rand(n_samples) * (r_max - r_min) + r_min
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)

def make_disjoint_circle(n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_half = n_samples // 2
        # Arc 1: -90 to 0 degrees
        theta1 = torch.rand(n_half) * (np.pi/2) - np.pi/2
        # Arc 2: 90 to 180 degrees
        theta2 = torch.rand(n_samples - n_half) * (np.pi/2) + np.pi/2

        theta = torch.cat([theta1, theta2])
        c = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        target_angle = theta * 4.0
        x_mean = torch.stack([torch.cos(target_angle), torch.sin(target_angle)], dim=1)
        x = x_mean + torch.randn_like(x_mean) * 0.05
        return x, c

def sampler(func: Callable, *args, **kwargs) -> Callable:
    return lambda n: func(n, *args, **kwargs)
