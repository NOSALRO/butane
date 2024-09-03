from typing import TypeAlias, Union, Optional, List
import math
import copy
import torch
from .._typedefs import *
from .._helpers import _fill_defaults, conv_def, _prod

class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32],
        *,
        activation_function: ModuleParams = [torch.nn.ReLU],
        conv_kernels: IntParams = [3],
        conv_stride: IntParams = [1],
        conv_pad: IntParams = [0],
        conv_bias: BoolParams = [True],
        conv_pad_mode: Optional[StrParams] = ['zeros'],
        pool: Optional[torch.nn.Module] = None,
        pool_kernels: IntParams = [0],
        pool_stride: IntParams = [1],
        pool_pad: IntParams = [0],
        dropout: FloatParams = [0.],
        output_activation: Optional[bool] = False,
        normalization: BoolParams = [False],
        normalization_type: Optional[torch.nn.Module] = None
    ) -> None:

        # Initialize default args.
        self.__input_dims = input_dims
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

        # Check if we are set to build our model
        assert (
            len(channels) == len(activation_function)
            and len(conv_kernels) == len(conv_stride)
            and len(conv_stride) == len(pool_kernels)
            and len(conv_stride) == len(conv_pad)
            and len(conv_pad) == len(conv_bias)
            and len(conv_bias) == len(pool_stride)
            and len(pool_kernels) == len(pool_stride)
            and len(pool_stride) == len(pool_pad)
            and len(pool_pad) == len(dropout)
            and len(dropout) == len(normalization)
        ), "Params size must be the same!"

        super().__init__()

        _channels = copy.copy(channels)
        _channels.insert(0, input_dims[0])

        self.conv_block = torch.nn.Sequential()
        for i in range(len(_channels) - 1):
            self.conv_block.extend(
                self.__create_subblock(
                    _channels[i],
                    _channels[i + 1],
                    conv_kernels[i],
                    conv_stride[i],
                    conv_pad[i],
                    conv_bias[i],
                    conv_pad_mode[i],
                    pool_kernels[i],
                    pool_stride[i],
                    pool_pad[i],
                    dropout[i],
                    activation_function[i] if (i + 1) != (len(_channels) - 1) or output_activation else None, # if the model is single Conv1dBlock check if we want activation function on the output
                    normalization[i], #if (i + 1) != (len(channels) - 1) else False, #
                    self.norm_type
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(torch.rand(1, *self.__input_dims)).size())
        self.train()
        return sz

    @staticmethod
    def __component_output_sz(component: torch.nn.Module, input_dim: List[int]) -> torch.Tensor:
        component.eval()
        with torch.no_grad():
            sz = torch.tensor(component(torch.rand(1, *input_dim)).size())
        component.train()
        return sz

    def __create_subblock(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel: int,
        conv_stride: int,
        conv_pad: int,
        conv_bias: bool,
        conv_pad_mode: str,
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
        pool = self.pool(pool_kernel, stride = pool_stride, padding = pool_pad) if _prod(pool_kernel) != 0 else None
        drop = torch.nn.Dropout(dropout) if dropout != 0 else None
        if norm_type.__name__ == "LayerNorm":
            norm_layer = norm_type(list(self.__component_output_sz(conv, self.output_size[1:])[1:]))
        else:
            norm_layer = norm_type(out_channels)
        ret.append(conv)
        if norm:
            ret.append(norm_layer)
        if af is not None:
            ret.append(af)
        if pool is not None:
            ret.append(pool)
        if drop is not None:
            ret.append(drop)
        return ret

class ConvTransposeBlock(torch.nn.Module):

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32, 64],
        *,
        activation_function: ModuleParams = [torch.nn.ReLU],
        conv_kernels: IntParams = [3],
        conv_stride: IntParams = [1],
        conv_pad: IntParams = [0],
        conv_bias: BoolParams = [True],
        conv_output_padding: IntParams = [0],
        pool: Optional[torch.nn.Module] = None,
        pool_kernels: IntParams = [0],
        pool_stride: IntParams = [1],
        pool_pad: IntParams = [0],
        dropout: FloatParams = [0],
        output_activation: Optional[bool] = False,
        normalization: BoolParams = [False],
        normalization_type: Optional[torch.nn.Module] = None
    ) -> None:

        # Initialize default args.
        self.__input_dims = input_dims
        if pool is not None:
            self.pool = pool

        if normalization_type is not None:
            self.norm_type = normalization_type

        activation_function = _fill_defaults(activation_function, len(channels))
        conv_kernels = _fill_defaults(conv_kernels, len(channels), self.N)
        conv_stride = _fill_defaults(conv_stride, len(channels), self.N)
        conv_pad = _fill_defaults(conv_pad, len(channels), self.N)
        conv_output_padding = _fill_defaults(conv_output_padding, len(channels), self.N)
        pool_kernels = _fill_defaults(pool_kernels, len(channels), self.N)
        pool_stride = _fill_defaults(pool_stride, len(channels), self.N)
        pool_pad = _fill_defaults(pool_pad, len(channels), self.N)
        dropout = _fill_defaults(dropout, len(channels))
        normalization = _fill_defaults(normalization, len(channels))
        conv_bias = _fill_defaults(conv_bias, len(channels))


        # Check if we are set to build our model
        assert (
            len(channels) == len(activation_function)
            and len(conv_kernels) == len(conv_stride)
            and len(conv_stride) == len(pool_kernels)
            and len(conv_stride) == len(conv_pad)
            and len(conv_pad) == len(conv_bias)
            and len(conv_bias) == len(pool_stride)
            and len(pool_kernels) == len(pool_stride)
            and len(pool_stride) == len(pool_pad)
            and len(pool_pad) == len(dropout)
            and len(dropout) == len(normalization)
        ), "Params size must be the same!"

        super().__init__()

        _channels = copy.copy(channels)
        _channels.insert(0, input_dims[0])

        self.conv_transpose_block = torch.nn.Sequential()
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
                    pool_kernels[i],
                    pool_stride[i],
                    pool_pad[i],
                    dropout[i],
                    activation_function[i] if (i + 1) != (len(_channels) - 1) or output_activation else None, # if the model is single Conv1dBlock check if we want activation function on the output
                    normalization[i], #if (i + 1) != (len(channels) - 1) else False, #
                    self.norm_type
            ))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose_block(x)

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(torch.rand(1, *self.__input_dims)).size())
        self.train()
        return sz

    @staticmethod
    def __component_output_sz(component: torch.nn.Module, input_dim: List[int]) -> torch.Tensor:
        component.eval()
        with torch.no_grad():
            sz = torch.tensor(component(torch.rand(1, *input_dim)).size())
        component.train()
        return sz

    def __create_subblock(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel: int,
        conv_stride: int,
        conv_pad: int,
        conv_bias: bool,
        conv_out_pad: str,
        pool_kernel: int,
        pool_stride: int,
        pool_pad: int,
        dropout: float,
        af: Optional[torch.nn.Module],
        norm: bool,
        norm_type: torch.nn.Module
    ) -> List[torch.nn.Module]:
        ret = []
        conv = self.conv(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad, output_padding=conv_out_pad, bias=conv_bias)
        pool = self.pool(pool_kernel, stride = pool_stride, padding = pool_pad) if _prod(pool_kernel) != 0 else None
        drop = torch.nn.Dropout(dropout) if dropout != 0 else None
        ret.append(conv)
        if norm:
            if norm_type.__name__ == "LayerNorm":
                norm_layer = norm_type(list(self.__component_output_sz(conv, self.output_size[1:])[1:]))
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

@conv_def('1d')
class Conv1dBlock(ConvBlock): ...

@conv_def('1d', True)
class ConvTranspose1dBlock(ConvTransposeBlock): ...

@conv_def('2d')
class Conv2dBlock(ConvBlock): ...

@conv_def('2d', True)
class ConvTranspose2dBlock(ConvTransposeBlock): ...

@conv_def('3d')
class Conv3dBlock(ConvBlock): ...

@conv_def('3d', True)
class ConvTranspose3dBlock(ConvTransposeBlock): ...
