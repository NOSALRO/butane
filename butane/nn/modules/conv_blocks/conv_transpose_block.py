from typing import Optional, List, Tuple
import copy
import torch

from ...._typedefs import *
from ...._helpers import _fill_defaults, _prod, module_name
from .conv_base_block import ConvBlockBase
from ._conv_utils import define_Nd_convolution


class ConvTransposeBlock(ConvBlockBase):

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32, 64],
        *,
        activation_function: ModuleParams = [torch.nn.ReLU()],
        conv_kernels: IntParams = [3],
        conv_stride: IntParams = [1],
        conv_pad: IntParams = [0],
        conv_bias: BoolParams = [True],
        conv_output_padding: IntParams = [0],
        dropout: FloatParams = [0],
        output_activation: Optional[bool] = False,
        normalization: BoolParams = [False],
        normalization_type: Optional[torch.nn.Module] = None
    ) -> None:

        # Initialize default args.
        super().__init__()
        self.input_dims = input_dims

        if normalization_type is not None:
            self.norm_type = normalization_type

        activation_function = _fill_defaults(activation_function, len(channels))
        conv_kernels = _fill_defaults(conv_kernels, len(channels), self.N)
        conv_stride = _fill_defaults(conv_stride, len(channels), self.N)
        conv_pad = _fill_defaults(conv_pad, len(channels), self.N)
        conv_output_padding = _fill_defaults(conv_output_padding, len(channels), self.N)
        dropout = _fill_defaults(dropout, len(channels))
        normalization = _fill_defaults(normalization, len(channels))
        conv_bias = _fill_defaults(conv_bias, len(channels))
        self.norm_type = _fill_defaults(self.norm_type, len(channels))


        # Check if we are set to build our model
        assert len(channels) == len(activation_function), "Params size must be the same!"
        assert len(channels) == len(conv_kernels), "Params size must be the same!"
        assert len(channels) == len(conv_stride), "Params size must be the same!"
        assert len(channels) == len(conv_pad), "Params size must be the same!"
        assert len(channels) == len(conv_bias), "Params size must be the same!"
        assert len(channels) == len(conv_output_padding), "Params size must be the same!"
        assert len(channels) == len(dropout), "Params size must be the same!"
        assert len(channels) == len(normalization), "Params size must be the same!"
        assert len(channels) == len(self.norm_type), "Params size must be the same!"

        _channels = copy.copy(channels)
        _channels.insert(0, input_dims[0])

        self.conv_transpose_block = torch.nn.ModuleList()
        for i in range(len(_channels) - 1):
            self.conv_transpose_block.extend(
                self.__create_subblock(
                    _channels[i],
                    _channels[i + 1],
                    conv_kernels[i],
                    conv_stride[i],
                    conv_pad[i],
                    conv_bias[i],
                    conv_output_padding[i],
                    dropout[i],
                    activation_function[i] if (i + 1) != (len(_channels) - 1) or output_activation else None, # if the model is single Conv1dBlock check if we want activation function on the output
                    normalization[i], #if (i + 1) != (len(channels) - 1) else False, #
                    self.norm_type[i]
            ))

    def module_list(self):
        if isinstance(self.conv_transpose_block, torch.nn.ModuleList):
            return self.conv_transpose_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.conv_transpose_block, torch.nn.ModuleList):
            x = self._forward_module_list(self.conv_transpose_block, x)
        elif isinstance(self.conv_transpose_block, torch.nn.Sequential):
            x = self._forward_sequential(self.conv_transpose_block, x)
        return x

    def sequential(self):
        self.conv_transpose_block = torch.nn.Sequential(*self.conv_transpose_block)
        return self

    def __iter__(self):
        for module in self.conv_transpose_block:
            yield module

    def __create_subblock(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel: Tuple[int, ...],
        conv_stride: Tuple[int, ...],
        conv_pad: Tuple[int, ...],
        conv_bias: bool,
        conv_out_pad: str,
        dropout: float,
        af: Optional[torch.nn.Module],
        norm: bool,
        norm_type: torch.nn.Module
    ) -> List[torch.nn.Module]:
        ret = []
        conv = self.conv(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad, output_padding=conv_out_pad, bias=conv_bias)
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
        if drop is not None:
            ret.append(drop)
        return ret

@define_Nd_convolution('1d', True)
class ConvTranspose1dBlock(ConvTransposeBlock): ...

@define_Nd_convolution('2d', True)
class ConvTranspose2dBlock(ConvTransposeBlock): ...

@define_Nd_convolution('3d', True)
class ConvTranspose3dBlock(ConvTransposeBlock): ...
