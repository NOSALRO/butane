from typing import TypeAlias, Union, Optional, List, Tuple, Callable
import torch
from ..._typedefs import *
from ..._helpers import module_name

def calculate_output_size(*modules, input_dims: IntParams) -> torch.Tensor:

    _input = torch.randn(1, *input_dims)
    out_sz = None
    for module in modules:
        if hasattr(module, 'output_size'):
            _input = torch.randn(1, *module.output_size)
            continue
        _input = module(_input)
    return torch.tensor(_input.size())[1:]

@torch.no_grad
def init_weights(model: torch.nn.Module, weight_init_method: Callable, bias_init_method: Optional[Callable] = None):
    print(model)
    for module in model.modules():
        if hasattr(module, 'weight'):
            if 'norm' in module._get_name().lower():
                continue
            if isinstance(module, torch.nn.Embedding):
                continue
            weight_init_method(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None and bias_init_method is not None:
                bias_init_method(module.bias.data)

class Unflatten(torch.nn.Module):
    def __init__(self, start_dim: int, sizes: torch.Tensor) -> None:
        super().__init__()
        sz = []
        for s in sizes:
            sz.append(int(s.item()))
        self.unflatten = torch.nn.Unflatten(1, sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unflatten(x)
