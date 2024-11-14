from typing import TypeAlias, Union, Optional, List, Tuple, Self
import math
import copy
import torch
from .._typedefs import *
from .._helpers import _fill_defaults, conv_def, _prod, module_name

class ConvBlockBase(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.input_dims = None

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            _input = torch.rand(1, *self.input_dims)
            for module in self.children():
                if not isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                    continue
                if isinstance(module, torch.nn.ModuleList):
                    for layer in module:
                        if hasattr(layer, 'input_dims'):
                            _input = torch.rand(1, *layer.input_dims)
                        _input = layer(_input)
                elif isinstance(module, torch.nn.Sequential):
                    _input = module(_input)
            sz = torch.tensor(_input.size()[1:])
        self.train()
        return sz

    @staticmethod
    def _component_output_sz(component: torch.nn.Module, input_dim: List[int]) -> torch.Tensor:
        component.eval()
        with torch.no_grad():
            sz = torch.tensor(component(torch.rand(1, *input_dim)).size())
        component.train()
        return sz

    def _forward_module_list(self, module: torch.nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for layer in module:
            x = layer(x)
        return x

    def _forward_sequential(self, module: torch.nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        x = module(x)
        return x

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
        if isinstance(self.conv_block, torch.nn.ModuleList):
            for layer in self.conv_block:
                x = layer(x)
        elif isinstance(self.conv_block, torch.nn.Sequential):
            x = self.conv_block(x)
        return x

    def sequential(self) -> Self:
        self.conv_block = torch.nn.Sequential(*self.conv_block)
        return self

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

@conv_def('1d')
class Conv1dBlock(ConvBlock): ...

@conv_def('2d')
class Conv2dBlock(ConvBlock): ...

@conv_def('3d')
class Conv3dBlock(ConvBlock): ...

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

@conv_def('1d', True)
class ConvTranspose1dBlock(ConvTransposeBlock): ...

@conv_def('2d', True)
class ConvTranspose2dBlock(ConvTransposeBlock): ...

@conv_def('3d', True)
class ConvTranspose3dBlock(ConvTransposeBlock): ...


class ConvUpsampleBlock(ConvBlockBase):

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
        super().__init__()
        self.input_dims = input_dims

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

@conv_def('1d')
class ConvUpsample1dBlock(ConvUpsampleBlock): ...

@conv_def('2d')
class ConvUpsample2dBlock(ConvUpsampleBlock): ...

@conv_def('3d')
class ConvUpsample3dBlock(ConvUpsampleBlock): ...


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

    def sequential(self) -> Self:
        self.conv_transpose_block = torch.nn.Sequential(*self.conv_transpose_block)
        return self

    def __iter__(self):
        for module in self.conv_transpose_block:
            yield module

@conv_def('1d')
class ConvTransposeToConv1dBlock(ConvTransposeToConvBlock): ...

@conv_def('2d')
class ConvTransposeToConv2dBlock(ConvTransposeToConvBlock): ...

@conv_def('3d')
class ConvTransposeToConv3dBlock(ConvTransposeToConvBlock): ...
