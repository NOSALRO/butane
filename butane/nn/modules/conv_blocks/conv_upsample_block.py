from typing import Optional, List, Tuple
import copy
import torch

from ...._typedefs import *
from ...._helpers import _fill_defaults, conv_def, _prod, module_name
from .conv_base_block import ConvBlockBase
from ._conv_utils import define_Nd_convolution


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

        self.conv_upsample_block = torch.nn.ModuleList()
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

    def module_list(self):
        if isinstance(self.conv_upsample_block, torch.nn.ModuleList):
            return self.conv_upsample_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.conv_upsample_block, torch.nn.ModuleList):
            x = self._forward_module_list(self.conv_upsample_block, x)
        elif isinstance(self.conv_upsample_block, torch.nn.Sequential):
            x = self._forward_sequential(self.conv_upsample_block, x)
        return x

    def sequential(self):
        self.conv_upsample_block = torch.nn.Sequential(*self.conv_upsample_block)
        return self

    def __iter__(self):
        for module in self.conv_upsample_block:
            yield module

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

@define_Nd_convolution('1d')
class ConvUpsample1dBlock(ConvUpsampleBlock): ...

@define_Nd_convolution('2d')
class ConvUpsample2dBlock(ConvUpsampleBlock): ...

@define_Nd_convolution('3d')
class ConvUpsample3dBlock(ConvUpsampleBlock): ...
