from typing import Callable, Optional, Union, Tuple
from functools import partial
import torch
from ..._typedefs import *
from ..modules.residual_blocks import *
from ..modules.conv_blocks import Conv1dBlock, Conv2dBlock, Conv3dBlock
from ..modules.mlp_block import MLPBlock
from ..modules.attention import LocalSelfAttention1d, LocalSelfAttention2d
from ..modules.embeddings import SinusoidalEmbeddings, LearnableEmbeddings
from ..utils import utils

class ResBlockNd(torch.nn.Module):
    residual_block_creator: torch.nn.Module
    conv_block: torch.nn.Module
    conv: torch.nn.Module

    def __init__(self, input_dims: IntParams, channels: int, embedding_size: Optional[int] = None):
        super().__init__()
        self.__input_dims = input_dims

        if embedding_size is not None:
            self.time_projection = MLPBlock(
                input_dims=embedding_size,
                output_dims=channels,
                hidden_dims=[embedding_size * 2],
                activation_function=[torch.nn.SiLU()],
            )

        self.pre_embedding = torch.nn.Sequential(
            torch.nn.GroupNorm(32, self.__input_dims[0]),
            torch.nn.SiLU(),
            self.conv(self.__input_dims[0], channels, 3, padding=1),
        )

        self.post_embedding = torch.nn.Sequential(
            torch.nn.GroupNorm(32, channels),
            torch.nn.SiLU(),
            utils.zero_module(self.conv(channels, channels, 3, padding=1)),
        )

        self.shortcut = self.conv(self.__input_dims[0], channels, 1) if self.__input_dims[0] != channels else torch.nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
    ) -> torch.Tensor:

        if t is not None:
            t = self.time_projection(t)
            while t.dim() != x.dim():
                t = t[..., None]

        h = self.pre_embedding(x)
        if t is not None:
            h = h + t
        h = self.post_embedding(h)
        return h + self.shortcut(x)

    @property
    @torch.no_grad()
    def output_size(self) -> torch.Tensor:
        _input = torch.randn(1, *self.__input_dims)
        _out = self(_input)
        _sz = torch.tensor(_out.size())[1:]
        return _sz

class ResBlock1d(ResBlockNd):
    residual_block_creator = Residual1dBlock
    conv_block = Conv1dBlock
    conv = torch.nn.Conv1d

class ResBlock2d(ResBlockNd):
    residual_block_creator = Residual2dBlock
    conv_block = Conv2dBlock
    conv = torch.nn.Conv2d

class ResBlock3d(ResBlockNd):
    residual_block_creator = Residual3dBlock
    conv_block = Conv3dBlock
    conv = torch.nn.Conv3d

class DownsampleNd(torch.nn.Module):

    conv: torch.nn.Module
    residual_block_creator: torch.nn.Module

    def __init__(self, input_dims: IntParams, channels: int, embedding_size: int, attention: Optional[torch.nn.Module] = None):
        super().__init__()

        self.__input_dims = input_dims

        self.residual_block1 = self.residual_block_creator(input_dims, channels, embedding_size)
        residual_input = utils.calculate_output_size(self.residual_block1, input_dims=self.__input_dims)

        if attention is not None:
            self.attn = attention(channels)

        self.residual_block2 = self.residual_block_creator(residual_input, channels, embedding_size)
        self.down = self.conv(channels, channels, 3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
    ) -> torch.Tensor:

        x = self.residual_block1(x, t)
        if hasattr(self, "attn"):
            x = self.attn(x)
        x = self.residual_block2(x, t)
        return self.down(x), x

    @property
    @torch.no_grad()
    def output_size(self) -> torch.Tensor:
        _input = torch.randn(1, *self.__input_dims)
        _out = self(_input)[0]
        _sz = torch.tensor(_out.size())[1:]
        return _sz

class Downsample1d(DownsampleNd):
    conv = torch.nn.Conv1d
    residual_block_creator = ResBlock1d

class Downsample2d(DownsampleNd):
    conv = torch.nn.Conv2d
    residual_block_creator = ResBlock2d

class Downsample3d(DownsampleNd):
    conv = torch.nn.Conv3d
    residual_block_creator = ResBlock3d


class UpsampleNd(torch.nn.Module):
    conv_transpose: torch.nn.Module
    residual_block_creator: torch.nn.Module

    def __init__(self, input_dims: IntParams, channels: int, embedding_size: int, attention: Optional[torch.nn.Module] = None):
        super().__init__()

        self.__input_dims = input_dims
        self.__channels = channels

        self.up = self.conv_transpose(self.__input_dims[0], channels, 4, stride=2, padding=1)
        residual_input = utils.calculate_output_size(self.up, input_dims=self.__input_dims)
        residual_input[0] = channels*2

        self.residual_block1 = self.residual_block_creator(
            residual_input,
            channels,
            embedding_size,
        )

        if attention is not None:
            self.attn = attention(channels)

        residual_input = self.residual_block1.output_size
        self.residual_block2 = self.residual_block_creator(
            residual_input,
            channels,
            embedding_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t: torch.Tensor = None,
    ) -> torch.Tensor:

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        x = self.residual_block1(x, t)
        if hasattr(self, "attn"):
            x = self.attn(x)
        x = self.residual_block2(x, t)
        return x

    @property
    @torch.no_grad()
    def output_size(self) -> torch.Tensor:
        _input = torch.randn(1, *self.__input_dims)
        _skip = torch.randn(1, self.__channels, *self.__input_dims[1:])
        _out = self(_input, _skip)
        _sz = torch.tensor(_out.size())[1:]
        return _sz

class Upsample1d(UpsampleNd):
    conv_transpose = torch.nn.ConvTranspose1d
    residual_block_creator = ResBlock1d

class Upsample2d(UpsampleNd):
    conv_transpose = torch.nn.ConvTranspose2d
    residual_block_creator = ResBlock2d

class Upsample3d(UpsampleNd):
    conv_transpose = torch.nn.ConvTranspose3d
    residual_block_creator = ResBlock3d

class UNetNd(torch.nn.Module):
    conv: torch.nn.Module
    conv_block: torch.nn.Module
    pool: torch.nn.Module
    norm_type: torch.nn.Module
    residual_block_creator: torch.nn.Module
    downsample: torch.nn.Module
    upsample: torch.nn.Module
    attention: torch.nn.Module
    N: int

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32, 64],
        time_embedding_size: int = 256,
        attention: bool = False,
        self_condition: bool = False,
        use_film: bool = True,
        n_classes: Optional[int] = None,
        n_middle_blocks: int = 2,
        condition_input_dims: IntParams = None,
        output_channels: Optional[int] = None,
    ):
        assert (not self_condition and n_classes is not None) or (self_condition and n_classes is None) or (not self_condition and n_classes is None), "You can use either self-condition or class-codition"
        super().__init__()
        self.__input_dims = input_dims
        self._channels = channels
        self._use_film = use_film
        self._self_condition = self_condition
        self._condition_input_dims = condition_input_dims if condition_input_dims is not None else self.__input_dims
        self._output_channels = output_channels if output_channels is not None else self.__input_dims[0]

        self.attention = partial(
            self.attention,
            kernel_size=1,
            n_heads=1,
            dropout_p=0.0,
            prenorm=partial(torch.nn.GroupNorm, num_groups=32),
            bias=True,
            zero_conv=True,
        )

        self.init_layer = torch.nn.Sequential(
            self.conv(self.__input_dims[0], self._channels[0], 3, padding=1),
            torch.nn.GroupNorm(32, self._channels[0]),
            torch.nn.SiLU(),
        )

        self.time_embeddings = SinusoidalEmbeddings(time_embedding_size)
        # self.time_embeddings = LearnableEmbeddings(time_embedding_size)
        self.time_projection = MLPBlock(
            input_dims=time_embedding_size,
            output_dims=time_embedding_size,
            hidden_dims=[time_embedding_size * 4],
            activation_function=[torch.nn.SiLU()],
        )

        if self_condition:
            _condition_conv = self.conv_block(
                input_dims=self._condition_input_dims,
                channels=[self._channels[0] * 2 if (use_film and self.N > 1) else self._channels[0]],
                activation_function=[torch.nn.SiLU()],
                output_activation=True,
                conv_pad=[1],
                normalization=[True],
                normalization_type=partial(torch.nn.GroupNorm, num_groups=32)
            )
            _condition_conv_out = _condition_conv.output_size
            # _condition_residual = self.residual_block_creator(
            #     input_dims=_condition_conv_out,
            #     channels=self._channels[0] * 2 if (use_film and self.N > 1) else self._channels[0],
            # )
            self.condition_projection = torch.nn.Sequential(
                # utils.zero_module(_condition_conv),
                _condition_conv,
                # _condition_residual
            )
            if self.N == 1:
                self.condition_projection.extend([
                    torch.nn.Flatten(1),
                    torch.nn.Linear(_condition_conv_out.prod(), self._channels[0] * 2 if use_film else self._channels[0])
                ])
            if self._use_film:
                self.init_layer.append(torch.nn.GroupNorm(32, self._channels[0]))

        if n_classes is not None:
            _condition_embeddings = torch.nn.Embedding(n_classes, time_embedding_size)
            self.condition_projection = torch.nn.Sequential(
                    _condition_embeddings,
                    # MLPBlock(
                    #     input_dims=time_embedding_size,
                    #     output_dims=time_embedding_size * 2 if self._use_film else time_embedding_size,
                    #     hidden_dims=[],
                    #     activation_function=[torch.nn.SiLU()]
                    # )
            )

        self.down_blocks = torch.nn.ModuleList()
        input_sz = utils.calculate_output_size(self.init_layer, input_dims=self.__input_dims)
        for ch in self._channels:
            self.down_blocks.append(
                self.downsample(
                    input_sz,
                    ch,
                    embedding_size=time_embedding_size,
                    attention=self.attention if attention else None,
                )
            )
            input_sz = self.down_blocks[-1].output_size

        self.middle_blocks = torch.nn.ModuleList()
        input_sz = input_sz

        for i in range(n_middle_blocks):
            self.middle_blocks.append(
                self.residual_block_creator(
                    input_sz,
                    input_sz[0],
                    embedding_size=time_embedding_size,
                )
            )
            if attention and (i + 1) < n_middle_blocks:
                self.middle_blocks.append(self.attention(input_sz[0]))

        self.up_blocks = torch.nn.ModuleList()
        for ch in reversed(self._channels):
            self.up_blocks.append(
                self.upsample(
                    input_sz,
                    ch,
                    embedding_size=time_embedding_size,
                    attention=self.attention if attention else None,
                )
            )
            input_sz = self.up_blocks[-1].output_size

        self.out_conv = torch.nn.Sequential(
            torch.nn.GroupNorm(32, input_sz[0]),
            torch.nn.SiLU(),
            utils.zero_module(self.conv(input_sz[0], self._output_channels, 3, padding=1))
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if t is not None:
            t = self.time_embeddings(t)
            t = self.time_projection(t)

        if c is not None and hasattr(self, "condition_projection"):
            c = self.condition_projection(c)
            if self._self_condition:
                while c.dim() != x.dim():
                    c = c[..., None]
            if self._use_film:
                gamma, beta = c.chunk(2, dim=1)

        x = self.init_layer(x)

        if c is not None and self._self_condition:
            x = gamma * x + beta if self._use_film else x + c
        elif c is not None and not self._self_condition:
            t = gamma * t + beta if self._use_film else t + c

        _skip_connection = []
        for down in self.down_blocks:
            x, _skip = down(x, t)
            _skip_connection.append(_skip)

        for mb in self.middle_blocks:
            x = mb(x, t)

        for up, skip in zip(self.up_blocks, reversed(_skip_connection)):
            x = up(x, skip, t)
        return self.out_conv(x)

class UNet1d(UNetNd):
    conv = torch.nn.Conv1d
    conv_block = Conv1dBlock
    pool = torch.nn.MaxPool1d
    norm_type = torch.nn.BatchNorm1d
    residual_block_creator = ResBlock1d
    downsample = Downsample1d
    upsample = Upsample1d
    attention = LocalSelfAttention1d
    N = 1

class UNet2d(UNetNd):
    conv = torch.nn.Conv2d
    conv_block = Conv2dBlock
    pool = torch.nn.MaxPool2d
    norm_type = torch.nn.BatchNorm2d
    residual_block_creator = ResBlock2d
    downsample = Downsample2d
    upsample = Upsample2d
    attention = LocalSelfAttention1d
    N = 2

class UNet3d(UNetNd):
    conv = torch.nn.Conv3d
    conv_block = Conv3dBlock
    pool = torch.nn.MaxPool3d
    norm_type = torch.nn.BatchNorm3d
    residual_block_creator = ResBlock3d
    downsample = Downsample3d
    upsample = Upsample3d
    attention = LocalSelfAttention2d
    N = 3
