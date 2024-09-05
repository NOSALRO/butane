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
class ScaledTanh(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self._alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.scaled_tanh(x, self._alpha)
