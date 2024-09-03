import torch

def gaussian(x: torch.Tensor) -> torch.Tensor:
    return (-x.square()).exp()

def squashing(x: torch.Tensor) -> torch.Tensor:
    return (9 / 8. * torch.sin(x)) + (1 / 8. * torch.sin(3. * x))
