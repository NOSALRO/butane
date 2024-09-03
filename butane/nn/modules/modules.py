import torch
from .. import functional

class Gaussian(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.gaussian(x)

class Squashing(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.squashing(x)

class Unflatten(torch.nn.Module):
    def __init__(self, start_dim, sizes) -> None:
        super().__init__()
        sz = []
        for s in sizes:
            sz.append(int(s.item()))
        self.unflatten = torch.nn.Unflatten(1, sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unflatten(x)