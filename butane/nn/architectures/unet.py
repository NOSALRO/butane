from typing import Callable, Optional, Union, Tuple
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
from ..modules.embeddings import SinusoidalEmbeddings, LearnableEmbeddings, FourierEmbeddings
from ..utils import utils
from ..wrapper import XDependent, XDependentSequential


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
        embedding_size: Optional[int],
        output_channels: int,
        upsample: bool = False,
        downsample: bool = False,
        dropout: float = 0.0,
        use_film: bool = True,
        use_shortcut_conv: bool = False,
    ):

        super().__init__()
        self._use_film = use_film

        self.input_layer = torch.nn.Sequential(
            torch.nn.GroupNorm(32, input_dims[0]),
            torch.nn.SiLU(),
        )
        self.block_1 = torch.nn.Sequential(
            self.conv(input_dims[0], output_channels, 3, padding=1)
        )

        if embedding_size is not None:
            self.embedding_layers = torch.nn.Linear(
                    embedding_size,
                    (output_channels * 2) if use_film else output_channels,
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

            if self._use_film:
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

class ConditionProjectionBlockNd(torch.nn.Module):
    conv: torch.nn.Module
    residual_block_creator: XDependent

    def __init__(
        self,
        input_dims: IntParams,
        embedding_size: int,
        channels: int,
        n_residual_blocks: int = 1,
        dropout: float = 0.0,
        attention: Optional[torch.nn.Module] = None,
        use_film: bool = True
    ) -> None:
        super().__init__()
        self._input_dims = input_dims
        self.blocks = torch.nn.ModuleList()

        _updated_input_dims = copy.deepcopy(self._input_dims)
        self.input_layer = self.conv(self._input_dims[0], channels, 3, padding=1)
        _updated_input_dims = utils.calculate_output_size(self.input_layer, input_dims=_updated_input_dims)

        for i in range(n_residual_blocks):
            self.blocks.append(self.residual_block_creator(
                input_dims=_updated_input_dims,
                embedding_size=embedding_size,
                dropout=dropout,
                output_channels=channels,
                use_film=use_film,
                downsample=True,
            ))
            _updated_input_dims = utils.calculate_output_size(self.blocks[-1], input_dims=_updated_input_dims)

            if attention is not None and (i + 1) != n_residual_blocks:
                self.blocks.append(attention(_updated_input_dims[0]))

        self.pre_projection_block = torch.nn.Sequential(
            torch.nn.GroupNorm(32, _updated_input_dims[0]),
            torch.nn.SiLU(),
            torch.nn.Flatten(1)
        )

        self.linear_projection = MLPBlock(
            input_dims=_updated_input_dims.prod(),
            output_dims=embedding_size * 2 if use_film else embedding_size,
            hidden_dims=[embedding_size],
            activation_function=[torch.nn.SiLU()],
            output_activation=True,
            zero_out=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for b in self.blocks:
            h = b(h)
        h = self.pre_projection_block(h)
        h = self.linear_projection(h)
        return h

class ConditionProjectionBlock1d(ConditionProjectionBlockNd):
    conv = torch.nn.Conv1d
    residual_block_creator = ResBlock1d

class ConditionProjectionBlock2d(ConditionProjectionBlockNd):
    conv = torch.nn.Conv2d
    residual_block_creator = ResBlock2d

class ConditionProjectionBlock3d(ConditionProjectionBlockNd):
    conv = torch.nn.Conv3d
    residual_block_creator = ResBlock3d

class ConvAdapterNd(torch.nn.Module):
    conv: torch.nn.Module
    N: int

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams,
    ) -> None:

        super().__init__()
        self._input_dims = input_dims
        self._channels = channels
        self._channels.insert(0, self._input_dims[0])

        self.adapters = torch.nn.ModuleList()
        for i in range(len(self._channels) - 1):
            adapter = torch.nn.Sequential(
                self.conv(self._channels[i], self._channels[i+1], 3, padding=1),
                torch.nn.SiLU(),
                self.conv(self._channels[i+1], self._channels[i+1], 1, padding=0)
            )
            self.adapters.append(adapter)

class ConvAdapter1d(ConvAdapterNd):
    conv = torch.nn.Conv1d
    N: 1

class ConvAdapter2d(ConvAdapterNd):
    conv = torch.nn.Conv2d
    N: 2

class ConvAdapter3d(ConvAdapterNd):
    conv = torch.nn.Conv3d
    N: 3

# TOOD: Fix input size handling
# The model does not handle odd input size; only 2^n
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
        output_channels: Optional[int] = None,
        dropout: float = 0.0,
        attention: bool = False,
        attention_channel_idx: IntParams = [],
        use_film: bool = True,
        n_heads: int = 4,
        n_middle_blocks: int = 2,
        resample_with_resblock: bool = False,
        conv_resample: bool = True,
        zero_conv: bool = True,
        attention_dropout: float = 0.0,
        scale_residual: bool = False,
        time_dependent: bool = True,
        time_embedding_size: Optional[int] = None,
        embedding_size: Optional[int] = None,
        embedder: Optional[torch.nn.Module] = None,
        learn_embeddings: bool = False,
        n_classes: Optional[int] = None,
        concat_condition: bool = False,
        project_condition: bool = False,
        condition_input_dims: Optional[IntParams] = None,
        condition_dropout: float = 0.,
        condition_n_residuals: int = 2,
        condition_attention: bool = False,
    ):
        super().__init__()
        self._input_dims = input_dims
        self._channels = channels
        self._dropout = dropout
        self._output_channels = output_channels if output_channels is not None else self._input_dims[0]

        self._use_film = use_film
        self._has_condition = project_condition or concat_condition or (n_classes is not None)
        self._time_dependent = time_dependent
        self._n_classes = n_classes

        self._condition_input_dims = condition_input_dims if condition_input_dims is not None else copy.copy(self._input_dims)
        self._resample_with_resblock = resample_with_resblock
        self._n_residual_blocks = n_residual_blocks

        self._concat_condition = concat_condition
        if self._concat_condition:
            self._input_dims[0] += self._condition_input_dims[0]

        if not attention:
            attention_channel_idx = []
        else:
            if len(attention_channel_idx) == 0:
                attention_channel_idx = list(range(len(self._channels)))

        self._attention_channel_idx = attention_channel_idx

        self.attention = partial(
            self.attention_block,
            kernel_size=1,
            n_heads=n_heads,
            dropout_p=attention_dropout,
            prenorm=partial(torch.nn.GroupNorm, num_groups=32),
            bias=True,
            zero_out=zero_conv,
        )

        if self._time_dependent:
            self._time_embedding_size = time_embedding_size if time_embedding_size is not None else channels[0]
            self._embedding_size = embedding_size if embedding_size is not None else self._time_embedding_size * 4

            embedder = FourierEmbeddings if embedder is None else embedder
            self.time_embedder = embedder(d_model=self._time_embedding_size, learnable=learn_embeddings)

            self.embedding_projection = MLPBlock(
                    input_dims=self._time_embedding_size,
                    output_dims=self._embedding_size,
                    hidden_dims=[self._embedding_size],
                    activation_function=[torch.nn.SiLU()],
                    output_activation=False,
            )
        else:
            self._time_embedding_size = None
            self._embedding_size = None

        if project_condition and self._n_classes is None:
            self.condition_projection = self.condition_block(
                input_dims=self._condition_input_dims,
                channels=self._channels[0],
                embedding_size=self._embedding_size,
                dropout=condition_dropout,
                n_residual_blocks=condition_n_residuals,
                attention=self.attention if condition_attention else None,
            )
        elif self._n_classes is not None:
            self.class_embedder = torch.nn.Embedding(self._n_classes, self._embedding_size)

        self.input_layer = self.conv(self._input_dims[0], self._channels[0], 3, padding=1)
        _updated_input_dims = utils.calculate_output_size(self.input_layer, input_dims=self._input_dims)
        _downsampling_channels = [self._channels[0]]
        self._resampled = []

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
                        use_film=self._use_film
                ))
                _updated_input_dims = utils.calculate_output_size(_subblock[-1], input_dims=_updated_input_dims)

                if i in self._attention_channel_idx:
                    _subblock.append(self.attention(_updated_input_dims[0]))

                self.downsample_blocks.append(_subblock)
                _downsampling_channels.append(int(_updated_input_dims[0]))
                self._resampled.append(False)
            if (i + 1) != len(self._channels):
                self.downsample_blocks.append(
                    XDependentSequential(self.residual_block_creator(
                        input_dims=_updated_input_dims,
                        embedding_size=self._embedding_size,
                        dropout=self._dropout,
                        output_channels=ch,
                        use_film=self._use_film,
                        downsample=True,
                    ) if self._resample_with_resblock
                    else self.downsample(_updated_input_dims, use_conv=conv_resample, output_channels=ch)
                ))
                self._resampled.append(True)
                _updated_input_dims = utils.calculate_output_size(self.downsample_blocks[-1][-1], input_dims=_updated_input_dims)
                _downsampling_channels.append(int(_updated_input_dims[0]))

        self.middle_blocks = torch.nn.ModuleList()
        for i in range(n_middle_blocks):
            _subblock = XDependentSequential()
            _subblock.append(
                self.residual_block_creator(
                    input_dims=_updated_input_dims,
                    embedding_size=self._embedding_size,
                    dropout=self._dropout,
                    output_channels=_updated_input_dims[0],
                    use_film=self._use_film))
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
                        use_film=self._use_film))
                _updated_input_dims = utils.calculate_output_size(_subblock[-1], input_dims=_updated_input_dims)

                if i in self._attention_channel_idx:
                    _subblock.append(self.attention(_updated_input_dims[0]))

                if i and j == self._n_residual_blocks:
                    _subblock.append(self.residual_block_creator(
                            input_dims=_updated_input_dims,
                            embedding_size=self._embedding_size,
                            dropout=self._dropout,
                            output_channels=ch,
                            use_film=self._use_film,
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
        # self.adapter = ConvAdapter2d(self._condition_input_dims, [32, *self._channels])

    def prepare_conditioning(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        emb = None
        if t is not None:
            assert self._time_dependent, "Time dependency is not enabled, but time was provided"
            emb = self.time_embedder(t)
            emb = self.embedding_projection(emb)

        if self._has_condition and c is None:
            raise ValueError("Conditioning is enabled but no condition `c` was provided.")
        elif not self._has_condition and c is not None:
            raise ValueError("Conditioning is disabled; but condition `c` was provided.")

        if self._n_classes is not None:
            assert len(c.shape) == 1
            self.class_embedder(c)
            emb = emb + c
            return x, emb, c

        if self._concat_condition:
            x = torch.cat([x, c], dim=1)

        if hasattr(self, 'condition_projection') and emb is not None:
            c = self.condition_projection(c)
            if self._use_film:
                c_gamma, c_beta = c.chunk(chunks=2, dim=1)
                emb = emb * (1 + c_gamma) + c_beta
            else:
                emb = emb + c
        return x, emb, c

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x, emb, c = self.prepare_conditioning(x, t, c)
        h = self.input_layer(x)

        # f = []
        # for a in self.adapter.adapters:
        #     c = a(c)
        #     f.append(c)

        # stage = 1
        # _skip_connection = [h + f[0]]
        _skip_connection = [h]
        for i, down in enumerate(self.downsample_blocks):
            h = down(h, emb)
            # if self._resampled[i]:
            #     r = f[stage]
            #     r = torch.nn.functional.interpolate(r, size=h.shape[2:])
            #     h = h + r
            #     stage+=1
            _skip_connection.append(h)

        for mb in self.middle_blocks:
            h = mb(h, emb)

        for up in self.upsample_blocks:
            _skip = _skip_connection.pop()
            h = torch.cat([h, _skip], dim=1)
            h = up(h, emb)

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
    condition_block = ConditionProjectionBlock1d
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
    condition_block = ConditionProjectionBlock2d
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
    condition_block = ConditionProjectionBlock3d
    dims = 3
