from typing import Optional, List
import copy
import torch

from ...._typedefs import *
from ...._helpers import _fill_defaults, _prod, module_name
from .conv_base_block import ConvBlockBase
from ._conv_utils import define_Nd_convolution


class ConvTransposeToConvBlock(ConvBlockBase):

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32, 64],
        *,
        conv_blocks: Optional[int] = 2,
        conv_input_channels: IntParams = None,
        conv_channels: NestedIntParams = [[32]],
        conv_transpose_kernels: IntParams = [3],
        conv_transpose_stride: IntParams = [1],
        conv_transpose_pad: IntParams = [0],
        conv_transpose_bias: BoolParams = [True],
        conv_transpose_output_padding: IntParams = [0],
        activation_function: NestedModuleParams = [[torch.nn.ReLU()]],
        conv_kernels: NestedIntParams = [[3]],
        conv_stride: NestedIntParams = [[1]],
        conv_pad: NestedIntParams = [[0]],
        conv_bias: NestedBoolParams = [[True]],
        conv_pad_mode: NestedStrParams = [['zeros']],
        dropout: NestedFloatParams = [[0]],
        output_activation: BoolParams = [False],
        normalization: NestedBoolParams = [[False]],
        normalization_type: ModuleParams = [None]
    ) -> None:

        # Initialize default args.
        super().__init__()
        self.input_dims = input_dims

        if normalization_type is not None:
            self.norm_type = normalization_type

        if self.N == 1:
            self.conv_transpose = torch.nn.ConvTranspose1d
            self.conv_block = Conv1dBlock
        elif self.N == 2:
            self.conv_transpose = torch.nn.ConvTranspose2d
            self.conv_block = Conv2dBlock
        elif self.N == 3:
            self.conv_transpose = torch.nn.ConvTranspose3d
            self.conv_block = Conv3dBlock

        if conv_input_channels is None:
            conv_input_channels = channels

        conv_transpose_kernels = _fill_defaults(conv_transpose_kernels, len(channels), self.N)
        conv_transpose_stride = _fill_defaults(conv_transpose_stride, len(channels), self.N)

        conv_transpose_pad = _fill_defaults(conv_transpose_pad, len(channels), self.N)
        conv_transpose_output_padding = _fill_defaults(conv_transpose_output_padding, len(channels), self.N)
        conv_transpose_bias = _fill_defaults(conv_transpose_bias, len(channels))

        conv_channels = [_fill_defaults(el, conv_blocks) for el in conv_channels]
        conv_channels = conv_channels * len(channels) if len(conv_channels) != len(channels) else conv_channels

        conv_kernels = [_fill_defaults(el, conv_blocks, self.N) for el in conv_kernels]
        conv_kernels = conv_kernels * len(channels) if len(conv_kernels) != len(channels) else conv_kernels

        conv_stride = [_fill_defaults(el, conv_blocks, self.N) for el in conv_stride]
        conv_stride = conv_pad * len(channels) if len(conv_stride) != len(channels) else conv_stride

        conv_pad = [_fill_defaults(el, conv_blocks, self.N) for el in conv_pad]
        conv_pad = conv_pad * len(channels) if len(conv_pad) != len(channels) else conv_pad

        activation_function = [_fill_defaults(el, conv_blocks) for el in activation_function]
        activation_function = activation_function * len(channels) if len(activation_function) != len(channels) else activation_function

        conv_pad_mode = [_fill_defaults(el, conv_blocks) for el in conv_pad_mode]
        conv_pad_mode = conv_pad_mode * len(channels) if len(conv_pad_mode) != len(channels) else conv_pad_mode

        dropout = [_fill_defaults(el, conv_blocks) for el in dropout]
        dropout = dropout * len(channels) if len(dropout) != len(channels) else dropout

        normalization = [_fill_defaults(el, conv_blocks) for el in normalization]
        normalization = normalization * len(channels) if len(normalization) != len(channels) else normalization

        conv_bias = [_fill_defaults(el, conv_blocks) for el in conv_bias]
        conv_bias = conv_bias * len(channels) if len(conv_bias) != len(channels) else conv_bias

        self.norm_type = [_fill_defaults(el, conv_blocks) for el in self.norm_type]
        self.norm_type = self.norm_type * len(channels) if len(self.norm_type) != len(channels) else self.norm_type

        # Check if we are set to build our model
        assert len(channels) == len(activation_function), "Params size must be the same!"
        assert len(channels) == len(conv_input_channels), "Params size must be the same!"
        assert len(channels) == len(conv_channels), "Params size must be the same!"
        assert len(channels) == len(conv_kernels), "Params size must be the same!"
        assert len(channels) == len(conv_stride), "Params size must be the same!"
        assert len(channels) == len(conv_pad), "Params size must be the same!"
        assert len(channels) == len(conv_bias), "Params size must be the same!"
        assert len(channels) == len(conv_pad_mode), "Params size must be the same!"

        assert all(conv_blocks == len(i) for i in conv_channels), "Params size must be the same!"
        assert all(conv_blocks == len(i) for i in conv_kernels), "Params size must be the same!"
        assert all(conv_blocks == len(i) for i in conv_stride), "Params size must be the same!"
        assert all(conv_blocks == len(i) for i in conv_bias), "Params size must be the same!"
        assert all(conv_blocks == len(i) for i in conv_pad_mode), "Params size must be the same!"

        assert len(channels) == len(conv_transpose_kernels), "Params size must be the same!"
        assert len(channels) == len(conv_transpose_stride), "Params size must be the same!"
        assert len(channels) == len(conv_transpose_pad), "Params size must be the same!"
        assert len(channels) == len(conv_transpose_bias), "Params size must be the same!"
        assert len(channels) == len(conv_transpose_output_padding), "Params size must be the same!"
        assert len(channels) == len(dropout), "Params size must be the same!"
        assert len(channels) == len(normalization), "Params size must be the same!"
        assert len(channels) == len(self.norm_type), "Params size must be the same!"

        _channels = copy.copy(channels)
        _channels.insert(0, input_dims[0])

        _conv_channels = copy.copy(conv_channels)
        for i in range(len(conv_channels)):
            _conv_channels[i].insert(0, conv_input_channels[i])

        self.conv_transpose_block = torch.nn.ModuleList()
        for i in range(len(_channels) - 1):
            self.conv_transpose_block.append(
                self.conv_transpose(
                    _channels[i],
                    _channels[i + 1],
                    kernel_size=conv_transpose_kernels[i],
                    stride=conv_transpose_stride[i],
                    padding=conv_transpose_pad[i],
                    output_padding=conv_transpose_output_padding[i],
                    bias=conv_transpose_bias[i]))
            self.conv_transpose_block.append(self.conv_block(
                input_dims = [_conv_channels[i][0], *self.output_size.numpy()[1:]],
                channels = _conv_channels[i][1:],
                activation_function = activation_function[i],
                conv_kernels = conv_kernels[i],
                conv_stride = conv_stride[i],
                conv_pad = conv_pad[i],
                conv_bias = conv_bias[i],
                conv_pad_mode = conv_pad_mode[i],
                dropout = dropout[i],
                normalization = normalization[i],
                normalization_type = self.norm_type[i]))

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

@define_Nd_convolution('1d')
class ConvTransposeToConv1dBlock(ConvTransposeToConvBlock): ...

@define_Nd_convolution('2d')
class ConvTransposeToConv2dBlock(ConvTransposeToConvBlock): ...

@define_Nd_convolution('3d')
class ConvTransposeToConv3dBlock(ConvTransposeToConvBlock): ...
