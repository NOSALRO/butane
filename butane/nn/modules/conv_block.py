from typing import TypeAlias, Union, Optional, List, Tuple
import math
import copy
import torch
from .._typedefs import *
from .._helpers import _fill_defaults, conv_def, _prod, module_name

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
        normalization_type: ModuleParams = None
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
                    self.pool[i],
                    pool_kernels[i],
                    pool_stride[i],
                    pool_pad[i],
                    dropout[i],
                    activation_function[i] if (i + 1) != (len(_channels) - 1) or output_activation else None, # if the model is single Conv1dBlock check if we want activation function on the output
                    normalization[i], #if (i + 1) != (len(channels) - 1) else False, #
                    self.norm_type[i]
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(torch.rand(1, *self.__input_dims)).size()[1:])
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
        if norm:
            if module_name(norm_type) == "LayerNorm":
                norm_layer = norm_type(list(self.__component_output_sz(conv, self.output_size)[1:]))
            elif module_name(norm_type) == "GroupNorm":
                comp_out = self.__component_output_sz(conv, self.output_size)
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
        dropout: FloatParams = [0],
        output_activation: Optional[bool] = False,
        normalization: BoolParams = [False],
        normalization_type: Optional[torch.nn.Module] = None
    ) -> None:

        # Initialize default args.
        self.__input_dims = input_dims

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
                    dropout[i],
                    activation_function[i] if (i + 1) != (len(_channels) - 1) or output_activation else None, # if the model is single Conv1dBlock check if we want activation function on the output
                    normalization[i], #if (i + 1) != (len(channels) - 1) else False, #
                    self.norm_type[i]
            ))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose_block(x)

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(torch.rand(1, *self.__input_dims)).size()[1:])
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
        if norm:
            if module_name(norm_type) == "LayerNorm":
                norm_layer = norm_type(list(self.__component_output_sz(conv, self.output_size)[1:]))
            elif module_name(norm_type) == "GroupNorm":
                comp_out = self.__component_output_sz(conv, self.output_size)
                norm_layer = norm_type(num_channels=comp_out[1])
            else:
                norm_layer = norm_type(out_channels)
            ret.append(norm_layer)
        if af is not None:
            ret.append(af)
        if drop is not None:
            ret.append(drop)
        return ret

class ConvUpsampleBlock(torch.nn.Module):

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
        conv_pad_mode: Optional[StrParams] = ['zeros'],
        upsample_size: IntParams = None,
        upsample_scale_factor: IntParams = [2],
        upsample_mode: StrParams = ["nearest"],
        upsample_align_corners: BoolParams = [False],
        dropout: FloatParams = [0],
        output_activation: Optional[bool] = False,
        normalization: BoolParams = [False],
        normalization_type: Optional[torch.nn.Module] = None
    ) -> None:

        # Initialize default args.
        self.__input_dims = input_dims

        if normalization_type is not None:
            self.norm_type = normalization_type

        activation_function = _fill_defaults(activation_function, len(channels))
        conv_kernels = _fill_defaults(conv_kernels, len(channels), self.N)
        conv_stride = _fill_defaults(conv_stride, len(channels), self.N)
        conv_pad = _fill_defaults(conv_pad, len(channels), self.N)
        conv_pad_mode = _fill_defaults(conv_pad_mode, len(channels))
        conv_bias = _fill_defaults(conv_bias, len(channels))
        upsample_size = _fill_defaults(upsample_size, len(channels), self.N) if upsample_size is not None else _fill_defaults([None], len(channels))
        upsample_scale_factor = _fill_defaults(upsample_scale_factor, len(channels), self.N)
        upsample_mode = _fill_defaults(upsample_mode, len(channels))
        upsample_align_corners = _fill_defaults(upsample_align_corners, len(channels))
        dropout = _fill_defaults(dropout, len(channels))
        normalization = _fill_defaults(normalization, len(channels))
        self.norm_type = _fill_defaults(self.norm_type, len(channels))

        # Check if we are set to build our model
        assert len(channels) == len(activation_function), "Params size must be the same!"
        assert len(channels) == len(conv_kernels), "Params size must be the same!"
        assert len(channels) == len(conv_stride), "Params size must be the same!"
        assert len(channels) == len(conv_pad), "Params size must be the same!"
        assert len(channels) == len(conv_bias), "Params size must be the same!"
        assert len(channels) == len(conv_pad_mode), "Params size must be the same!"
        assert len(channels) == len(dropout), "Params size must be the same!"
        assert len(channels) == len(upsample_scale_factor), "Params size must be the same!"
        assert len(channels) == len(upsample_size), "Params size must be the same!"
        assert len(channels) == len(upsample_mode), "Params size must be the same!"
        assert len(channels) == len(upsample_align_corners), "Params size must be the same!"
        assert len(channels) == len(normalization), "Params size must be the same!"
        assert len(channels) == len(self.norm_type), "Params size must be the same!"

        super().__init__()

        _channels = copy.copy(channels)
        _channels.insert(0, input_dims[0])

        self.conv_upsample_block = torch.nn.Sequential()
        for i in range(len(_channels) - 1):
            self.conv_upsample_block.extend(
                self.__create_subblock(
                    _channels[i],
                    _channels[i + 1],
                    conv_kernels[i],
                    conv_stride[i],
                    conv_pad[i],
                    conv_bias[i],
                    conv_pad_mode[i],
                    upsample_size[i],
                    upsample_scale_factor[i],
                    upsample_mode[i],
                    upsample_align_corners[i],
                    dropout[i],
                    activation_function[i] if (i + 1) != (len(_channels) - 1) or output_activation else None, # if the model is single Conv1dBlock check if we want activation function on the output
                    normalization[i], #if (i + 1) != (len(channels) - 1) else False, #
                    self.norm_type[i]
            ))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_upsample_block(x)

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(torch.rand(1, *self.__input_dims)).size()[1:])
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
        conv_kernel: Tuple[int, ...],
        conv_stride: Tuple[int, ...],
        conv_pad: Tuple[int, ...],
        conv_bias: bool,
        conv_pad_mode: str,
        upsample_size: Tuple[int, ...],
        upsample_scale_factor: int,
        upsample_mode: int,
        upsample_align_corners: bool,
        dropout: float,
        af: Optional[torch.nn.Module],
        norm: bool,
        norm_type: torch.nn.Module
    ) -> List[torch.nn.Module]:
        ret = []
        align_corners_modes = ['linear', 'cubic']
        if upsample_size is None:
            if upsample_mode in align_corners_modes:
                upsample = torch.nn.Upsample(scale_factor=upsample_scale_factor, mode=upsample_mode, align_corners=upsample_align_corners)
            else:
                upsample = torch.nn.Upsample(scale_factor=upsample_scale_factor, mode=upsample_mode)
        else:
            if upsample_mode in align_corners_modes:
                upsample = torch.nn.Upsample(size=upsample_size, mode=upsample_mode, align_corners=upsample_align_corners)
            else:
                upsample = torch.nn.Upsample(size=upsample_size, mode=upsample_mode)
        conv = self.conv(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad, padding_mode=conv_pad_mode, bias=conv_bias)
        drop = torch.nn.Dropout(dropout) if dropout != 0 else None
        ret.append(upsample)
        ret.append(conv)
        if norm:
            if module_name(norm_type) == "LayerNorm":
                norm_layer = norm_type(list(self.__component_output_sz(conv, self.output_size)[1:]))
            elif module_name(norm_type) == "GroupNorm":
                comp_out = self.__component_output_sz(conv, self.output_size)
                norm_layer = norm_type(num_channels=comp_out[1])
            else:
                norm_layer = norm_type(out_channels)
            ret.append(norm_layer)
        if af is not None:
            ret.append(af)
        if drop is not None:
            ret.append(drop)
        return ret

@conv_def('1d')
class Conv1dBlock(ConvBlock): ...

@conv_def('1d', True)
class ConvTranspose1dBlock(ConvTransposeBlock): ...

@conv_def('1d')
class ConvUpsample1dBlock(ConvUpsampleBlock): ...

@conv_def('2d')
class Conv2dBlock(ConvBlock): ...

@conv_def('2d', True)
class ConvTranspose2dBlock(ConvTransposeBlock): ...

@conv_def('2d')
class ConvUpsample2dBlock(ConvUpsampleBlock): ...

@conv_def('3d')
class Conv3dBlock(ConvBlock): ...

@conv_def('3d', True)
class ConvTranspose3dBlock(ConvTransposeBlock): ...

@conv_def('3d')
class ConvUpsample3dBlock(ConvUpsampleBlock): ...
