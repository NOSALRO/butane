from abc import abstractmethod
import torch

class XDependent(torch.nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor, *args): ...

class XDependentSequential(torch.nn.Sequential, XDependent):
    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, XDependent):
                x = layer(x, *args)
            else:
                x = layer(x)
        return x

