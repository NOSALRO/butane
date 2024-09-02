import math
import copy
from typing import TypeAlias, Union, Optional
import torch
import numpy as np
from .._typedefs import *

def ConvDef(conv_type, transpose=False):
    def inner(cls):
        if conv_type == '1d':
            cls.conv = torch.nn.Conv1d if not transpose else torch.nn.ConvTranspose1d
            cls.pool = torch.nn.MaxPool1d
            cls.norm_type = torch.nn.BatchNorm1d
        elif conv_type == '2d':
            cls.conv = torch.nn.Conv2d if not transpose else torch.nn.ConvTranspose2d
            cls.pool = torch.nn.MaxPool2d
            cls.norm_type = torch.nn.BatchNorm2d
        elif conv_type == '3d':
            cls.conv = torch.nn.Conv3d if not transpose else torch.nn.ConvTranspose3d
            cls.pool = torch.nn.MaxPool3d
            cls.norm_type = torch.nn.BatchNorm3d
        return cls
    return inner


class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32, 64],
        *,
        activation_function: ModuleParams = None,
        conv_kernels: IntParams = None,
        conv_stride: IntParams = None,
        conv_pad: IntParams = None,
        conv_bias: BoolParams = True,
        conv_pad_mode: Optional[str] = 'zeros',
        pool: Optional[torch.nn.Module] = None,
        pool_kernels: IntParams = None,
        pool_stride: IntParams = None,
        pool_pad: IntParams = None,
        dropout: FloatParams = None,
        output_activation: Optional[bool] = False,
        normalization: BoolParams = False,
        normalization_type: Optional[torch.nn.Module] = None
    ) -> None:

        # Initialize default args.
        self.__input_dims = input_dims
        if pool is not None:
            self.pool = pool

        if normalization_type is not None:
            self.norm_type = normalization_type

        if activation_function is None:
            activation_function = [torch.nn.ReLU() for _ in range(len(channels))]

        if conv_kernels is None:
            conv_kernels = [3 for _ in range(len(channels))]

        if conv_stride is None:
            conv_stride = [1 for _ in range(len(channels))]

        if conv_pad is None:
            conv_pad = [0 for _ in range(len(channels))]

        if pool_kernels is None:
            pool_kernels = [0 for _ in range(len(channels))]

        if pool_stride is None:
            pool_stride = [1 for _ in range(len(channels))]

        if pool_pad is None:
            pool_pad = [0 for _ in range(len(channels))]

        if dropout is None:
            dropout = [0. for _ in range(len(channels))]

        if isinstance(normalization, bool):
            normalization = [normalization for _ in range(len(channels))]

        if isinstance(conv_bias, bool):
            conv_bias = [conv_bias for _ in range(len(channels))]

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

        # assert (output_activation is None or isinstance(output_activation, torch.nn.Module)), "Ouput actviation function must be a torch.nn.Module"

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
                    conv_pad_mode,
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
    def output_size(self):
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(torch.rand(1, *self.__input_dims)).size())
        self.train()
        return sz

    @staticmethod
    def __component_output_sz(component, input_dim):
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
        af: Optional[torch.nn.Module] = None,
        norm: Optional[bool] = False,
        norm_type: Optional[torch.nn.Module] = torch.nn.BatchNorm1d
    ) -> list:
        ret = []
        conv = self.conv(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad, padding_mode=conv_pad_mode, bias=conv_bias)
        pool = self.pool(pool_kernel, stride = pool_stride, padding = pool_pad) if np.prod(pool_kernel) != 0 else None
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
        activation_function: ModuleParams = None,
        conv_kernels: IntParams = None,
        conv_stride: IntParams = None,
        conv_pad: IntParams = None,
        conv_bias: BoolParams = True,
        conv_output_padding: IntParams = None,
        pool: Optional[torch.nn.Module] = None,
        pool_kernels: IntParams = None,
        pool_stride: IntParams = None,
        pool_pad: IntParams = None,
        dropout: FloatParams = None,
        output_activation: Optional[bool] = False,
        normalization: BoolParams = False,
        normalization_type: Optional[torch.nn.Module] = torch.nn.BatchNorm1d
    ) -> None:

        # Initialize default args.
        self.__input_dims = input_dims
        if pool is not None:
            self.pool = pool

        if normalization_type is not None:
            self.norm_type = normalization_type

        if activation_function is None:
            activation_function = [torch.nn.ReLU() for _ in range(len(channels))]

        if conv_kernels is None:
            conv_kernels = [3 for _ in range(len(channels))]

        if conv_stride is None:
            conv_stride = [1 for _ in range(len(channels))]

        if conv_pad is None:
            conv_pad = [0 for _ in range(len(channels))]

        if conv_output_padding is None:
            conv_output_padding = [0 for _ in range(len(channels))]

        if pool_kernels is None:
            pool_kernels = [0 for _ in range(len(channels))]

        if pool_stride is None:
            pool_stride = [1 for _ in range(len(channels))]

        if pool_pad is None:
            pool_pad = [0 for _ in range(len(channels))]

        if dropout is None:
            dropout = [0. for _ in range(len(channels))]

        if isinstance(normalization, bool):
            normalization = [normalization for _ in range(len(channels))]

        if isinstance(conv_bias, bool):
            conv_bias = [conv_bias for _ in range(len(channels))]

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

        # assert (output_activation is None or isinstance(output_activation, torch.nn.Module)), "Ouput actviation function must be a torch.nn.Module"

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
    def output_size(self):
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(torch.rand(1, *self.__input_dims)).size())
        self.train()
        return sz

    @staticmethod
    def __component_output_sz(component, input_dim):
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
        af: Optional[torch.nn.Module] = None,
        norm: Optional[bool] = False,
        norm_type: Optional[torch.nn.Module] = torch.nn.BatchNorm1d
    ) -> list:
        ret = []
        conv = self.conv(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad, output_padding=conv_out_pad, bias=conv_bias)
        pool = self.pool(pool_kernel, stride = pool_stride, padding = pool_pad) if np.prod(pool_kernel) != 0 else None
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

@ConvDef('1d')
class Conv1dBlock(ConvBlock): ...

@ConvDef('1d', True)
class ConvTranspose1dBlock(ConvTransposeBlock): ...

@ConvDef('2d')
class Conv2dBlock(ConvBlock): ...

@ConvDef('2d', True)
class ConvTranspose2dBlock(ConvTransposeBlock): ...

@ConvDef('3d')
class Conv3dBlock(ConvBlock): ...

@ConvDef('3d', True)
class ConvTranspose3dBlock(ConvTransposeBlock): ...
