from abc import abstractmethod
from typing import Any
import torch

class XDependent(torch.nn.Module):
    @abstractmethod
    def forward(self, x: Any, *args, **kwargs) -> Any: 
        ...

class XDependentSequential(torch.nn.Sequential, XDependent):
    def forward(self, x: Any, *args, **kwargs) -> Any:
        for layer in self:
            if isinstance(layer, XDependent):
                x = layer(x, *args, **kwargs)
            else:
                x = layer(x)
        return x
