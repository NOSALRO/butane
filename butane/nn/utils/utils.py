from typing import TypeAlias, Union, Optional, List, Tuple
import torch
from .._typedefs import *

def calculate_output_size(*modules, input_dims: IntParams):

    _input = torch.randn(1, *input_dims)
    out_sz = None
    for module in modules:
        if hasattr(module, 'output_size'):
            _input = torch.randn(1, *module.output_size)
            continue
        _input = module(_input)
    return torch.tensor(_input.size())[1:]
