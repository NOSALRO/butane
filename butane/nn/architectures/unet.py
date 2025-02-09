import math
import copy
from typing import Optional, Callable, Tuple
from ..modules import *
from ..._typedefs import *
import torch

def get_double_conv(in_channels: int, out_channels: int) -> torch.nn.Module:
    return Conv1dBlock(
        input_dims = in_channels,
        channels = [out_channels, out_channels],
        activation_function = [torch.nn.ReLU()],
        conv_kernels = [3],
        conv_stride = [1],
        conv_pad = [1],
        conv_bias = [False],
        pool_kernels = [0, 0],
        pool_stride = [0, 2],
        normalization = [True],
        output_activation = True
    )

class DownsampleBlock(torch.nn.Module):
    def __init__(
        self,
        input_dims: IntParams,
        out_channels: int,
        *,
        init: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.__input_dims = input_dims
        if not init:
            self.downsample_block = torch.nn.Sequential(
                    torch.nn.MaxPool1d(2),
                    get_double_conv(self.__input_dims, out_channels)
                )
        else:
            self.downsample_block = get_double_conv(self.__input_dims, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample_block(x)

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(torch.rand(1, *self.__input_dims)).size()[1:])
        self.train()
        return sz

class UpsampleBlock(torch.nn.Module):
    def __init__(
        self,
        input_dims: IntParams,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.__input_dims = input_dims
        self.upsample = torch.nn.ConvTranspose1d(self.__input_dims[0].item(), self.__input_dims[0].item() // 2, kernel_size=2, stride=2)
        self.conv_block = get_double_conv(self.__input_dims, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            sz = torch.tensor(self.forward(
                torch.rand(1, *self.__input_dims), 
                torch.rand(1, self.__input_dims[0]//2, *(self.__input_dims[1:] * 2))
            ).size()[1:])
        self.train()
        return sz

class UNet1d(torch.nn.Module):
    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32], 
        *,
        scale_factor: Optional[int] = None,
        n_blocks: Optional[int] = 5
    ):
        super().__init__()
        self.__input_dims = input_dims
        self.__channels = copy.deepcopy(channels)

        if scale_factor:
            _scale_factors = [pow(scale_factor, i) for i in range(0, n_blocks, scale_factor - 1)]
            self.__channels = [self.__channels[0] * f for f in _scale_factors]

        self.downsample = torch.nn.ModuleList()
        for i in range(len(self.__channels)):
            if not i:
                self.downsample.append(DownsampleBlock(self.__input_dims, self.__channels[i], init=True))
            else:
                self.downsample.append(DownsampleBlock(self.downsample[-1].output_size, self.__channels[i]))

        self.upsample = torch.nn.ModuleList()
        for i in range(len(self.__channels)-1):
            self.upsample.append(UpsampleBlock(self.downsample[-1].output_size if not i else self.upsample[-1].output_size, self.__channels[-(i+2)]))
        self.output_conv = torch.nn.Conv1d(self.upsample[-1].output_size[0], self.__input_dims[0], kernel_size=1)

    def forward(self, x):
        x_downsampled = []
        for downsample_block in self.downsample:
            x_downsampled.append(downsample_block(x_downsampled[-1]) if len(x_downsampled) else downsample_block(x))

        for i, upsample_block in enumerate(self.upsample):
            x = upsample_block(x_downsampled[-1], x_downsampled[-2]) if not i else upsample_block(x, x_downsampled[-(i+2)])
        return self.output_conv(x)
