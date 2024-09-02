import torch
from .. import functional

class Gaussian(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return functional.gaussian(x)

class Squashing(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return functional.squashing(x)


class Unflatten(torch.nn.Module):
    def __init__(self, start_dim, sizes):
        super().__init__()
        sz = []
        for s in sizes:
            sz.append(int(s.item()))
        self.unflatten = torch.nn.Unflatten(1, sz)

    def forward(self, x):
        return self.unflatten(x)