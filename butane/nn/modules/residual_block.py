from typing import TypeAlias, Union, Optional, List, Tuple
import math
import copy
import torch
from .._typedefs import *
from .._helpers import _fill_defaults, conv_def, _prod
from .conv_block import *
from functools import partial

def conv_def(conv_type: str):
    def inner(cls):
        if conv_type == '1d':
            cls.conv = Conv1dBlock
        elif conv_type == '2d':
            cls.conv = Conv2dBlock
        elif conv_type == '3d':
            cls.conv = Conv3dBlock
        return cls
    return inner

class ResidualBlock(torch.nn.Module):

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32, 64],
        *,
        activation_function: torch.nn.Module = torch.nn.ReLU(),
        conv_kernels: Optional[int] = 3,
        conv_stride: Optional[int] = 1,
        conv_pad: Optional[int] = 1,
        conv_bias: Optional[bool] = True,
        conv_pad_mode: Optional[str] = 'zeros',
        pool: Optional[torch.nn.Module] = None,
        pool_kernels: int = 0,
        pool_stride: int = 1,
        pool_pad: int = 0,
        dropout: float = 0.,
        output_activation: Optional[bool] = False,
        normalization: BoolParams = [False],
        normalization_type: Optional[torch.nn.Module] = None,
        shortcut_normalization: Optional[bool] = False,
        shortcut_normalization_type: Optional[torch.nn.Module] = None
    ):
        super().__init__()

        normalization = _fill_defaults(normalization, 2)

        self.pool = None
        self.conv_module = self.conv(
            input_dims = input_dims,
            channels = [channels, channels],
            activation_function = [activation_function],
            conv_stride = [conv_stride, 1],
            conv_pad = [conv_pad, conv_pad],
            conv_bias = [conv_bias, conv_bias],
            conv_pad_mode = [conv_pad_mode, conv_pad_mode],
            pool_kernels = [0, 0],
            dropout = [0, dropout],
            output_activation = output_activation,
            normalization_type = [normalization_type],
            normalization = [normalization[0], normalization[1]]
        )

        self.shortcut = torch.nn.Identity()
        if conv_stride != 1 or input_dims[0] != channels:
            self.shortcut = self.conv(
                input_dims = input_dims,
                channels = [channels],
                activation_function = [torch.nn.Identity()],
                conv_kernels = [1],
                conv_stride = [conv_stride],
                conv_pad = [0],
                conv_bias = [False],
                normalization = shortcut_normalization,
                normalization_type = shortcut_normalization_type,
            )

        if pool is not None:
            self.pool = pool(kernel_size=pool_kernels, stride=pool_stride, padding=pool_pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_module(x)
        out += self.shortcut(x)
        if self.pool is not None:
            out = self.pool(out)
        return out

@conv_def('1d')
class Residual1dBlock(ResidualBlock): ...

@conv_def('2d')
class Residual2dBlock(ResidualBlock): ...

@conv_def('3d')
class Residual3dBlock(ResidualBlock): ...
