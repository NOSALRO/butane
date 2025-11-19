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
    SelfAttention,
    LocalSelfAttention1d,
    LocalSelfAttention2d,
    LocalCrossAttention1d,
    LocalCrossAttention2d
)
from ..modules.embeddings import SinusoidalEmbeddings, LearnableEmbeddings, PatchEmbeddings, FourierEmbeddings
from ..utils import utils


class DiTBlock(torch.nn.Module):

    def __init__(
        self,
        input_dims: int,
        output_ratio: float,
        embedding_size: Optional[int] = None,
        num_heads: int = 8,
    ) -> None:

        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = int(input_dims * output_ratio)
        self._embedding_size = embedding_size if embedding_size is not None else input_dims

        self.norm_1 = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False)
        self.norm_2 = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False)
        self.attn = SelfAttention(self._input_dims, n_heads=num_heads)
        self.mlp = MLPBlock(
            input_dims=self._input_dims,
            output_dims=self._input_dims,
            hidden_dims=[self._hidden_dims],
            activation_function=[torch.nn.GELU(approximate='tanh')],
            output_activation=False,
        )
        self.adaLN = torch.nn.Sequential(
            torch.nn.SiLU(),
            utils.zero_module(torch.nn.Linear(self._embedding_size, 6 * self._input_dims))
        )

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaLN(e).chunk(chunks=6, dim=1)

        x = x + alpha_1.unsqueeze(1) * self.attn(self.norm_1(x) * (1 + gamma_1.unsqueeze(1)) + beta_1.unsqueeze(1))
        x = x + alpha_2.unsqueeze(1) * self.mlp(self.norm_2(x) * (1 + gamma_2.unsqueeze(1)) + beta_2.unsqueeze(1))
        return x

class OutputBlock(torch.nn.Module):

    def __init__(
        self,
        input_dims: int,
        output_channels: int,
        patch_size: int = 1,
        embedding_size: Optional[int] = None
    ) -> None:

        super().__init__()
        self._input_dims = input_dims
        self._output_channels = output_channels
        self._embedding_size = embedding_size if embedding_size is not None else input_dims

        self.norm = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False)
        self.fc1 = utils.zero_module(torch.nn.Linear(self._input_dims, patch_size * patch_size * output_channels))
        self.adaLN = torch.nn.Sequential(
            torch.nn.SiLU(),
            utils.zero_module(torch.nn.Linear(self._embedding_size, 2 * self._input_dims))
        )

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.adaLN(e).chunk(chunks=2, dim=1)
        x = self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.fc1(x)
        return x

class DiT(torch.nn.Module):

    def __init__(
        self,
        input_dims: IntParams,
        hidden_dims: int = 1152,
        patch_size: int = 2,
        output_channels: Optional[int] = None,
        depth: int = 2,
        time_embedding_size: int = 256,
        embedding_size: Optional[int] = None,
        embedder: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._patch_size = patch_size
        self._output_channels = self._input_dims[0] if output_channels is None else output_channels

        self._time_embedding_size = time_embedding_size
        embedder = FourierEmbeddings if embedder is None else embedder
        self.time_embedder = embedder(d_model=self._time_embedding_size)
        self.embedding_projection = MLPBlock(
                input_dims=self._time_embedding_size,
                output_dims=self._hidden_dims,
                hidden_dims=[self._hidden_dims],
                activation_function=[torch.nn.SiLU()],
                output_activation=False,
                bias=[True]
        )

        self._num_of_patches = int((self._input_dims[1] / self._patch_size) * (self._input_dims[2] / self._patch_size))

        self._2d_embeddings = torch.nn.Parameter(self._get_2d_positional_embeddings(self._hidden_dims, n_patches=self._num_of_patches).float().unsqueeze(0), requires_grad=False)
        self._2d_embeddings_condition = torch.nn.Parameter(self._get_2d_positional_embeddings(self._hidden_dims, n_patches=self._num_of_patches).float().unsqueeze(0), requires_grad=False)

        self.patchify = PatchEmbeddings(
            self._input_dims,
            patch_size=self._patch_size,
            d_model=self._hidden_dims,
            bias=True,
        )

        self.patchify_condition = PatchEmbeddings(
            self._input_dims,
            patch_size=self._patch_size,
            d_model=self._hidden_dims,
            bias=True,
        )

        self.dit_blocks = torch.nn.ModuleList()
        for _ in range(depth):
            self.dit_blocks.append(DiTBlock(self._hidden_dims, 4.))

        self.output_layer = OutputBlock(self._hidden_dims, self._output_channels, patch_size=self._patch_size)
        self.initialize_weights()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        emb = self.embedding_projection(self.time_embedder(t))

        x = self.patchify(x) + self._2d_embeddings

        if c is not None:
            c = self.patchify_condition(c) + self._2d_embeddings_condition
            x = x + c

        for b in self.dit_blocks:
            x = b(x, emb)
        x = self.output_layer(x, emb)
        x = self._reshape(x)
        return x

    def _get_2d_positional_embeddings(self, d_model: int, n_patches: int) -> torch.Tensor:
        xx = torch.arange(0, n_patches**0.5)
        yy = torch.arange(0, n_patches**0.5)
        grid = torch.meshgrid(xx, yy, indexing='xy')
        grid = torch.stack(grid, dim=0).unsqueeze(1)
        emb_H = FourierEmbeddings.get_embeddings(grid[0].reshape(-1), d_model//2)
        emb_W = FourierEmbeddings.get_embeddings(grid[1].reshape(-1), d_model//2)
        _2d_embbendings = torch.cat([emb_H, emb_W], dim=-1)
        return _2d_embbendings

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        utils.init_weights(
            self.embedding_projection, 
            weight_init_method=torch.nn.init.xavier_uniform_,
            bias_init_method=partial(torch.nn.init.constant_, val=0)
        )

        utils.init_weights(
            self.embedding_projection, 
            weight_init_method=partial(torch.nn.init.normal_, std=0.02)
        )

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        H = W = int(math.sqrt(x.size(1)))
        x = x.view(x.size(0), H, W, self._patch_size, self._patch_size, self._output_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(x.size(0), self._output_channels, self._patch_size * H, self._patch_size * W)
        return x
