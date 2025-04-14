import math
import copy
from typing import Optional, Callable, Tuple
import torch

from ..modules import *
from ..._typedefs import *

def define_Nd_unet(conv_type: str, transpose: Optional[bool] = False) -> Callable[object, object]:
    def inner(cls):
        if conv_type == '1d':
            cls.conv = torch.nn.Conv1d
            cls.conv_block_creator = Conv1dBlock
            cls.up_conv_block_creator = ConvTransposeWRefinement1dBlock
            cls.pool = torch.nn.MaxPool1d
            cls.N = 1
        elif conv_type == '2d':
            cls.conv = torch.nn.Conv2d
            cls.conv_block_creator = Conv2dBlock
            cls.up_conv_block_creator = ConvTransposeWRefinement2dBlock
            cls.pool = torch.nn.MaxPool2d
            cls.N = 2
        elif conv_type == '3d':
            cls.conv = torch.nn.Conv3d
            cls.conv_block_creator = Conv3dBlock
            cls.up_conv_block_creator = ConvTransposeWRefinement3dBlock
            cls.pool = torch.nn.MaxPool3d
            cls.N = 3
        return cls
    return inner


class UNet(torch.nn.Module):

    def __init__(
        self,
        input_dims: IntParams,
        channels: int,
        n_blocks: int,
        expand_factor: int,
        n_mid_blocks: int = 1
    ):
        super().__init__()

        self.__input_dims = input_dims
        self._expand_by = 1

        self.down_blocks = torch.nn.ModuleList()
        self.mid_blocks = torch.nn.ModuleList()
        self.up_blocks = torch.nn.ModuleList()

        for i in range(n_blocks):
            block_input_dims = self.__input_dims if i == 0 else self.down_blocks[-1].output_size
            self.down_blocks.append(self._down_double_conv(block_input_dims, channels * self._expand_by))
            self._expand_by *= expand_factor

        for i in range(n_mid_blocks):
            block_input_dims = self.down_blocks[-1].output_size if i == 0 else self.mid_blocks[-1].output_size
            self.mid_blocks.append(self._down_double_conv(block_input_dims, channels * self._expand_by, pooling=False))

        for i in range(n_blocks):
            block_input_dims = self.mid_blocks[-1].output_size if i == 0 else self.up_blocks[-1].output_size
            self._expand_by = int(self._expand_by/expand_factor)
            self.up_blocks.append(self._up_double_conv(block_input_dims, channels * self._expand_by))

        self.out_proj = self.conv(self.up_blocks[-1].output_size[0], self.__input_dims[0], kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        down_blocks_outputs = []
        for i, down_block in enumerate(self.down_blocks):
            db = down_block.module_list()
            for l in db[:-1]:
                x = l(x)
            down_blocks_outputs.append(x)
            x = db[-1](x)

        for mid_block in self.mid_blocks:
            x = mid_block(x)

        for i, up_block in enumerate(self.up_blocks):
            x = up_block[0](x)
            x, skip = self.__padding(x, down_blocks_outputs[-(i+1)])
            x = torch.cat([skip, x], dim=1)
            x = up_block[1](x)

        x = self.out_proj(x)
        return x

    def __padding(self, x1, x2):
        x1_last_dims = x1.shape[2:]
        x2_last_dims = x2.shape[2:]

        pad = []
        for i in range(len(x1_last_dims)):
            diff = x2_last_dims[i] - x1_last_dims[i]
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad.extend([pad_before, pad_after])
        x1 = torch.nn.functional.pad(x1, pad)
        return x1, x2

    def _down_double_conv(
        self,
        input_dims: IntParams,
        channels: int,
        *,
        pooling: Optional[bool] = True
    ) -> torch.nn.Module:

        return self.conv_block_creator(
            input_dims = input_dims,
            channels = [channels, channels],
            activation_function = [torch.nn.ReLU()],
            conv_stride = [1, 1],
            conv_bias = [True],
            conv_pad = [0],
            output_activation = True,
            pool = self.pool,
            pool_kernels = [0, 2] if pooling else [0, 0],
            pool_stride = [0, 2]
        )

    def _up_double_conv(
        self,
        input_dims: IntParams,
        channels: int,
    ) -> torch.nn.Module:

        return self.up_conv_block_creator(
            input_dims = input_dims,
            channels = [channels],
            conv_transpose_kernels = [2],
            conv_transpose_pad = [0],
            conv_blocks = 2,
            conv_input_channels = [channels * 2],
            conv_channels = [[channels, channels]],
            conv_kernels = [[2, 2]],
            conv_stride = [[1, 2]],
            conv_bias = [True],
            conv_pad = [0],
            activation_function = [torch.nn.ReLU()],
            output_activation = [True],
        )


@define_Nd_unet('1d')
class UNet1d(UNet): ...

@define_Nd_unet('2d')
class UNet2d(UNet): ...

@define_Nd_unet('3d')
class UNet3d(UNet): ...
