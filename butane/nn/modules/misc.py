
import torch

class Unflatten(torch.nn.Module):
    def __init__(self, start_dim: int, sizes: torch.Tensor) -> None:
        super().__init__()
        sz = []
        for s in sizes:
            sz.append(int(s.item()))
        self.unflatten = torch.nn.Unflatten(1, sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unflatten(x)