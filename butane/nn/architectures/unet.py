from typing import Callable, Optional, Union, Tuple
from abc import abstractmethod
from functools import partial
import copy
import math
import torch
from ..._typedefs import *
from ..modules.residual_blocks import *
from ..modules.conv_blocks import Conv1dBlock, Conv2dBlock, Conv3dBlock
from ..modules.mlp_block import MLPBlock
from ..modules.attention import (
    LocalSelfAttention1d,
    LocalSelfAttention2d,
    LocalCrossAttention1d,
    LocalCrossAttention2d
)
from ..modules.embeddings import SinusoidalEmbeddings, LearnableEmbeddings
from ..utils import utils

class SelfConditionNd(torch.nn.Module):
    conv: torch.nn.Module

class SelfCondition1d(SelfConditionNd):
    conv = torch.nn.Conv1d

class SelfCondition2d(SelfConditionNd):
    conv = torch.nn.Conv2d

class SelfCondition3d(SelfConditionNd):
    conv = torch.nn.Conv3d

class XDependent(torch.nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor, e: torch.Tensor): ...


class XDependentSequential(torch.nn.Sequential, XDependent):
    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, XDependent):
                x = layer(x, e)
            else:
                x = layer(x)
        return x

class DownsampleNd(torch.nn.Module):
    conv: torch.nn.Module
    pool: torch.nn.Module
    dims: int

    def __init__(
        self,
        input_dims: IntParams,
        output_channels: Optional[int] = None,
        use_conv: bool = False
    ):
        super().__init__()
        if output_channels is None:
            output_channels = input_dims[0]

        stride = 2 if self.dims != 3 else (1, 2, 2)
        if use_conv:
            self.downsample_layer = self.conv(input_dims[0], output_channels, 3, stride=stride, padding=1)
        else:
            self.downsample_layer = self.pool(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample_layer(x)

class Downsample1d(DownsampleNd):
    conv = torch.nn.Conv1d
    pool = torch.nn.AvgPool1d
    dims = 1

class Downsample2d(DownsampleNd):
    conv = torch.nn.Conv2d
    pool = torch.nn.AvgPool2d
    dims = 2

class Downsample3d(DownsampleNd):
    conv = torch.nn.Conv3d
    pool = torch.nn.AvgPool3d
    dims = 3

class UpsampleNd(torch.nn.Module):
    conv: torch.nn.Module
    dims: int

    def __init__(
        self,
        input_dims: IntParams,
        output_channels: Optional[int] = None,
        refine: bool = False
    ):
        super().__init__()
        if output_channels is None:
            output_channels = input_dims[0]

        if self.dims == 3:
            self.upsample_layer = torch.nn.Upsample(size=(input_dims[1], input_dims[2] * 2, input_dims[3] * 2), mode='nearest')
        else:
            self.upsample_layer = torch.nn.Upsample(scale_factor=2, mode='nearest')

        if refine:
            self.refinement_layer = self.conv(input_dims[0], output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.upsample_layer(x)
        if hasattr(self, 'refinement_layer'):
            h = self.refinement_layer(h)
        return h

class Upsample1d(UpsampleNd):
    conv = torch.nn.Conv1d
    dims = 1

class Upsample2d(UpsampleNd):
    conv = torch.nn.Conv2d
    dims = 2

class Upsample3d(UpsampleNd):
    conv = torch.nn.Conv3d
    dims = 3

class ResBlockNd(XDependent):
    conv: torch.nn.Module
    upsample_block: torch.nn.Module
    downsample_block: torch.nn.Module

    def __init__(
        self,
        input_dims: IntParams,
        embedding_size: int,
        output_channels: int,
        upsample: bool = False,
        downsample: bool = False,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        use_shortcut_conv: bool = False,
    ):

        super().__init__()

        self.input_layer = torch.nn.Sequential(
            torch.nn.GroupNorm(32, input_dims[0]),
            torch.nn.SiLU(),
        )
        self.block_1 = torch.nn.Sequential(
            self.conv(input_dims[0], output_channels, 3, padding=1)
        )

        self.embedding_layers = torch.nn.Linear(
                embedding_size,
                (output_channels * 2) if use_scale_shift_norm else output_channels,
        )

        if upsample:
            self.up_down_x = self.upsample_block(input_dims, refine=False)
            self.up_down_h = self.upsample_block(input_dims, refine=False)
        elif downsample:
            self.up_down_x = self.downsample_block(input_dims, use_conv=False)
            self.up_down_h = self.downsample_block(input_dims, use_conv=False)
        else:
            self.up_down_x = torch.nn.Identity()
            self.up_down_h = torch.nn.Identity()

        self.block_2 = torch.nn.Sequential(
            torch.nn.GroupNorm(32, output_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout)
        )

        self.output_layer = utils.zero_module(self.conv(output_channels, output_channels, kernel_size=3, padding=1))

        if output_channels == input_dims[0]:
            self.shortcut = torch.nn.Identity()
        elif use_shortcut_conv:
            self.shortcut = self.conv(input_dims[0], output_channels, 3, padding=1)
        else:
            self.shortcut = self.conv(input_dims[0], output_channels, 1)

    def forward(self, x: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_x = self.input_layer(x)
        h_x = self.up_down_h(h_x)
        x = self.up_down_x(x)
        h_x = self.block_1(h_x)

        if e is not None:
            h_e = self.embedding_layers(e).type(h_x.dtype)
            while h_x.dim() != h_e.dim():
                h_e = h_e[..., None]

            if h_x.size(1) == 2*h_e.size(1):
                scale, shift = torch.chunk(h_e, chunks=2, dim=1)
                h_x = self.block_2(h_x) * (1 + scale) + shift
                h_x = self.output_layer(h_x)
            elif h_x.size(1) == h_e.size(1):
                h_x = h_x + h_e
                h_x = self.output_layer(self.block_2(h_x))
        else:
            h_x = self.output_layer(self.block_2(h_x))
        return self.shortcut(x) + h_x

class ResBlock1d(ResBlockNd):
    conv = torch.nn.Conv1d
    upsample_block = Upsample1d
    downsample_block = Downsample1d

class ResBlock2d(ResBlockNd):
    conv = torch.nn.Conv2d
    upsample_block = Upsample2d
    downsample_block = Downsample2d

class ResBlock3d(ResBlockNd):
    conv = torch.nn.Conv3d
    upsample_block = Upsample3d
    downsample_block = Downsample3d

class UNetNd(torch.nn.Module):
    conv: torch.nn.Module
    conv_block: torch.nn.Module
    pool: torch.nn.Module
    norm_type: torch.nn.Module
    residual_block_creator: torch.nn.Module
    downsample: torch.nn.Module
    upsample: torch.nn.Module
    attention: torch.nn.Module
    condition_block: torch.nn.Module
    dims: int

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32, 64],
        n_residual_blocks: int = 2,
        dropout: float = 0.0,
        attention: bool = False,
        attention_channel_idx: IntParams = [],
        self_condition: bool = False,
        use_film: bool = True,
        use_scale_shift_norm: bool = True,
        n_heads: int = 4,
        n_classes: Optional[int] = None,
        n_middle_blocks: int = 2,
        condition_input_dims: Optional[IntParams] = None,
        output_channels: Optional[int] = None,
        resample_with_resblock: bool = False,
        conv_resample: bool = False,
        zero_conv: bool = True,
        concat_condition: bool = False,
        attention_dropout: float = 0.0,
        scale_residual: bool = False,
        time_embedding_size: Optional[int] = None,
        embedding_size: Optional[int] = None,
        embedder: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self._input_dims = input_dims
        self._channels = channels
        self._dropout = dropout
        self._output_channels = output_channels if output_channels is not None else self._input_dims[0]

        self._use_film = use_film
        self._use_scale_shift_norm = use_scale_shift_norm

        self._self_condition = self_condition
        self._condition_input_dims = condition_input_dims if condition_input_dims is not None else copy.copy(self._input_dims)
        self._concat_condition = concat_condition

        if self._concat_condition:
            self._input_dims[0] += self._condition_input_dims[0]

        if not attention:
            attention_channel_idx = []
        else:
            if len(attention_channel_idx) == 0:
                attention_channel_idx = list(range(1, len(self._channels) + 1))

        self._attention_channel_idx = attention_channel_idx
        self._n_residual_blocks = n_residual_blocks

        self._time_embedding_size = time_embedding_size if time_embedding_size is not None else channels[0] * 4
        self._embedding_size = embedding_size if embedding_size is not None else self._time_embedding_size

        self._resample_with_resblock = resample_with_resblock

        embedder = SinusoidalEmbeddings if embedder is None else embedder
        self.time_embedder = embedder(d_model=self._time_embedding_size)

        self.embedding_projection = MLPBlock(
                input_dims=self._time_embedding_size,
                output_dims=self._embedding_size,
                hidden_dims=[self._embedding_size],
                activation_function=[torch.nn.SiLU()],
                output_activation=True,
        )

        self.attention = partial(
            self.attention_block,
            kernel_size=1,
            n_heads=n_heads,
            dropout_p=attention_dropout,
            prenorm=partial(torch.nn.GroupNorm, num_groups=32),
            bias=True,
            zero_conv=zero_conv,
        )

        self.input_layer = self.conv(self._input_dims[0], self._channels[0], 3, padding=1)
        _updated_input_dims = utils.calculate_output_size(self.input_layer, input_dims=self._input_dims)
        _downsampling_channels = [self._channels[0]]

        self.downsample_blocks = torch.nn.ModuleList([])
        for i, ch in enumerate(self._channels):
            for _ in range(self._n_residual_blocks):
                _subblock = XDependentSequential()
                _subblock.append(
                    self.residual_block_creator(
                        input_dims=_updated_input_dims,
                        embedding_size=self._embedding_size,
                        dropout=self._dropout,
                        output_channels=ch,
                        use_scale_shift_norm=self._use_scale_shift_norm
                ))
                _updated_input_dims = utils.calculate_output_size(_subblock[-1], input_dims=_updated_input_dims)

                if i in self._attention_channel_idx:
                    _subblock.append(self.attention(_updated_input_dims[0]))

                self.downsample_blocks.append(_subblock)
                _downsampling_channels.append(_updated_input_dims[0].item())
                if (i + 1) != len(self._channels):
                    self.downsample_blocks.append(
                        XDependentSequential(self.residual_block_creator(
                            input_dims=_updated_input_dims,
                            embedding_size=self._embedding_size,
                            dropout=self._dropout,
                            output_channels=ch,
                            use_scale_shift_norm=self._use_scale_shift_norm,
                            downsample=True,
                        ) if self._resample_with_resblock
                        else self.downsample(_updated_input_dims, use_conv=conv_resample, output_channels=ch)
                    ))
                    _updated_input_dims = utils.calculate_output_size(self.downsample_blocks[-1][-1], input_dims=_updated_input_dims)
                    _downsampling_channels.append(_updated_input_dims[0].item())

        self.middle_blocks = torch.nn.ModuleList()
        for i in range(n_middle_blocks):
            _subblock = XDependentSequential()
            _subblock.append(
                self.residual_block_creator(
                    input_dims=_updated_input_dims,
                    embedding_size=self._embedding_size,
                    dropout=self._dropout,
                    output_channels=_updated_input_dims[0],
                    use_scale_shift_norm=self._use_scale_shift_norm))
            if (i + 1) != n_middle_blocks:
                _subblock.append(self.attention(_updated_input_dims[0]))
            self.middle_blocks.append(_subblock)

        self.upsample_blocks = torch.nn.ModuleList([])
        for i, ch in reversed(list(enumerate(self._channels))):
            for j in range(self._n_residual_blocks + 1):
                _subblock = []
                _updated_input_dims[0] = _updated_input_dims[0] + _downsampling_channels.pop()
                _subblock.append(
                    self.residual_block_creator(
                        input_dims=_updated_input_dims,
                        embedding_size=self._embedding_size,
                        dropout=self._dropout,
                        output_channels=ch,
                        use_scale_shift_norm=self._use_scale_shift_norm))
                _updated_input_dims = utils.calculate_output_size(_subblock[-1], input_dims=_updated_input_dims)

                if i in self._attention_channel_idx:
                    _subblock.append(self.attention(_updated_input_dims[0]))

                if i and j == self._n_residual_blocks:
                    _subblock.append(self.residual_block_creator(
                            input_dims=_updated_input_dims,
                            embedding_size=self._embedding_size,
                            dropout=self._dropout,
                            output_channels=ch,
                            use_scale_shift_norm=self._use_scale_shift_norm,
                            upsample=True,
                        ) if self._resample_with_resblock
                        else self.upsample(_updated_input_dims, refine=conv_resample, output_channels=ch)
                    )
                    _updated_input_dims = utils.calculate_output_size(_subblock[-1], input_dims=_updated_input_dims)
                self.upsample_blocks.append(XDependentSequential(*_subblock))

        self.output_block = torch.nn.Sequential(
            torch.nn.GroupNorm(32, _updated_input_dims[0]),
            torch.nn.SiLU(),
            utils.zero_module(self.conv(_updated_input_dims[0], self._output_channels, 3, padding=1))
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if t is not None:
            emb = self.time_embedder(t)
            emb = self.embedding_projection(emb)

        h = self.input_layer(x)

        _skip_connection = [h]
        for down in self.downsample_blocks:
            h = down(h, emb)
            _skip_connection.append(h)

        for mb in self.middle_blocks:
            h = mb(h, emb)

        for up in self.upsample_blocks:
            h = torch.cat([h, _skip_connection.pop()], dim=1)
            h = up(h, emb)

        # if self._concat_condition:
        #     x = torch.cat([x, r], dim=1)
        #     x = self._concated_condition_projection(x)

        return self.output_block(h)


class UNet1d(UNetNd):
    conv = torch.nn.Conv1d
    conv_block = Conv1dBlock
    pool = torch.nn.MaxPool1d
    norm_type = torch.nn.BatchNorm1d
    residual_block_creator = ResBlock1d
    downsample = Downsample1d
    upsample = Upsample1d
    attention_block = LocalSelfAttention1d
    condition_block = SelfCondition1d
    dims = 1

class UNet2d(UNetNd):
    conv = torch.nn.Conv2d
    conv_block = Conv2dBlock
    pool = torch.nn.MaxPool2d
    norm_type = torch.nn.BatchNorm2d
    residual_block_creator = ResBlock2d
    downsample = Downsample2d
    upsample = Upsample2d
    attention_block = LocalSelfAttention1d
    condition_block = SelfCondition2d
    dims = 2

class UNet3d(UNetNd):
    conv = torch.nn.Conv3d
    conv_block = Conv3dBlock
    pool = torch.nn.MaxPool3d
    norm_type = torch.nn.BatchNorm3d
    residual_block_creator = ResBlock3d
    downsample = Downsample3d
    upsample = Upsample3d
    attention_block = LocalSelfAttention2d
    condition_block = SelfCondition3d
    dims = 3
