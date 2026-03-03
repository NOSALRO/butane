from typing import Optional, List
import copy
import torch

from ...._typedefs import *
from ...._helpers import _fill_defaults, _prod, module_name
from .conv_base_block import ConvBlockBase
from ._conv_utils import define_Nd_convolution


class ConvBlock(ConvBlockBase):

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32],
        *,
        activation_function: ModuleParams = [torch.nn.ReLU()],
        conv_kernels: IntParams = [3],
        conv_stride: IntParams = [1],
        conv_pad: IntParams = [0],
        conv_bias: BoolParams = [True],
        conv_pad_mode: Optional[StrParams] = ['zeros'],
        pool: Optional[ModuleParams] = None,
        pool_kernels: IntParams = [0],
        pool_stride: IntParams = [1],
        pool_pad: IntParams = [0],
        dropout: FloatParams = [0.],
        output_activation: Optional[bool] = False,
        normalization: BoolParams = [False],
        normalization_type: ModuleParams = None
    ) -> None:

        super().__init__()

        # Initialize default args.
        self.input_dims = input_dims
        if pool is not None:
            self.pool = pool

        if normalization_type is not None:
            self.norm_type = normalization_type

        conv_bias = _fill_defaults(conv_bias, len(channels))
        activation_function = _fill_defaults(activation_function, len(channels))
        conv_kernels = _fill_defaults(conv_kernels, len(channels), self.N)
        conv_stride = _fill_defaults(conv_stride, len(channels), self.N)
        conv_pad = _fill_defaults(conv_pad, len(channels), self.N)
        conv_pad_mode = _fill_defaults(conv_pad_mode, len(channels))
        pool_kernels = _fill_defaults(pool_kernels, len(channels), self.N)
        pool_stride = _fill_defaults(pool_stride, len(channels), self.N)
        pool_pad = _fill_defaults(pool_pad, len(channels), self.N)
        dropout = _fill_defaults(dropout, len(channels))
        normalization = _fill_defaults(normalization, len(channels))
        self.norm_type = _fill_defaults(self.norm_type, len(channels))
        self.pool = _fill_defaults(self.pool, len(channels))

        # Check if we are set to build our model
        assert len(channels) == len(activation_function), "Params size must be the same!"
        assert len(channels) == len(conv_kernels), "Params size must be the same!"
        assert len(channels) == len(conv_stride), "Params size must be the same!"
        assert len(channels) == len(conv_pad), "Params size must be the same!"
        assert len(channels) == len(conv_bias), "Params size must be the same!"
        assert len(channels) == len(conv_pad_mode), "Params size must be the same!"
        assert len(channels) == len(self.pool), "Params size must be the same!"
        assert len(channels) == len(pool_kernels), "Params size must be the same!"
        assert len(channels) == len(pool_stride), "Params size must be the same!"
        assert len(channels) == len(pool_kernels), "Params size must be the same!"
        assert len(channels) == len(pool_pad), "Params size must be the same!"
        assert len(channels) == len(dropout), "Params size must be the same!"
        assert len(channels) == len(normalization), "Params size must be the same!"
        assert len(channels) == len(self.norm_type), "Params size must be the same!"

        _channels = copy.copy(channels)
        _channels.insert(0, input_dims[0])

        self.conv_block = torch.nn.ModuleList()
        for i in range(len(_channels) - 1):
            is_last = (i == len(channels) - 1)
            current_af = activation_function[i] if (not is_last or output_activation) else None

            self.conv_block.extend(
                self.__create_subblock(
                  in_channels=_channels[i],
                  out_channels=_channels[i + 1],
                  conv_kernel=conv_kernels[i],
                  conv_stride=conv_stride[i],
                  conv_pad=conv_pad[i],
                  conv_bias=conv_bias[i],
                  conv_pad_mode=conv_pad_mode[i],
                  pool=self.pool[i],
                  pool_kernel=pool_kernels[i],
                  pool_stride=pool_stride[i],
                  pool_pad=pool_pad[i],
                  dropout=dropout[i],
                  af=current_af,
                  norm=normalization[i],
                  norm_type=self.norm_type[i]
            ))

    def module_list(self):
        if isinstance(self.conv_block, torch.nn.ModuleList):
            return self.conv_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.conv_block, torch.nn.ModuleList):
            x = self._forward_module_list(self.conv_block, x)
        elif isinstance(self.conv_block, torch.nn.Sequential):
            x = self._forward_sequential(self.conv_block, x)
        return x

    def sequential(self):
        self.conv_block = torch.nn.Sequential(*self.conv_block)
        return self

    def __iter__(self):
        for module in self.conv_block:
            yield module

    def __create_subblock(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel: int,
        conv_stride: int,
        conv_pad: int,
        conv_bias: bool,
        conv_pad_mode: str,
        pool: torch.nn.Module,
        pool_kernel: int,
        pool_stride: int,
        pool_pad: int,
        dropout: float,
        af: torch.nn.Module,
        norm: bool,
        norm_type: torch.nn.Module,
    ) -> List[torch.nn.Module]:
        ret = []
        conv = self.conv(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad, padding_mode=conv_pad_mode, bias=conv_bias)
        pool = pool(pool_kernel, stride = pool_stride, padding = pool_pad) if _prod(pool_kernel) != 0 else None
        drop = torch.nn.Dropout(dropout) if dropout != 0 else None
        ret.append(conv)
        if norm and module_name(norm_type):
            if module_name(norm_type) == "LayerNorm":
                norm_layer = norm_type(list(self._component_output_sz(conv, self.output_size)[1:]))
            elif module_name(norm_type) == "GroupNorm":
                comp_out = self._component_output_sz(conv, self.output_size)
                norm_layer = norm_type(num_channels=comp_out[1])
            else:
                norm_layer = norm_type(out_channels)
            ret.append(norm_layer)
        if af is not None:
            ret.append(af)
        if pool is not None:
            ret.append(pool)
        if drop is not None:
            ret.append(drop)
        return ret

@define_Nd_convolution('1d')
class Conv1dBlock(ConvBlock): ...

@define_Nd_convolution('2d')
class Conv2dBlock(ConvBlock): ...

@define_Nd_convolution('3d')
class Conv3dBlock(ConvBlock): ...
