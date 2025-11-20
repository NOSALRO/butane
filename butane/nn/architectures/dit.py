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
    CrossAttention,
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
        adaLN_zero_path: bool = True,
        cross_attention_context: bool = False
    ) -> None:

        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = int(input_dims * output_ratio)
        self._embedding_size = embedding_size if embedding_size is not None else input_dims
        self._cross_attention_context = cross_attention_context
        self._adaLN_zero_path = adaLN_zero_path

        self.norm_pre_sa = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False, eps=1e-06)
        self.norm_pre_mlp = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False, eps=1e-06)
        self.attn = SelfAttention(self._input_dims, n_heads=num_heads, return_attention=True)
        self.mlp = MLPBlock(
            input_dims=self._input_dims,
            output_dims=self._input_dims,
            hidden_dims=[self._hidden_dims],
            activation_function=[torch.nn.GELU(approximate='tanh')],
            output_activation=False,
        )

        if self._adaLN_zero_path:
            self.adaLN = torch.nn.Sequential(
                torch.nn.SiLU(),
                utils.zero_module(torch.nn.Linear(self._embedding_size, 6 * self._input_dims))
            )
        else:
            self.film = torch.nn.Sequential(
                torch.nn.SiLU(),
                utils.zero_module(torch.nn.Linear(self._embedding_size, 4 * self._input_dims))
            )

        if self._cross_attention_context:
            self.cattn_context_norm = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False)
            self.cattn_context_module = CrossAttention(self._input_dims, n_heads=num_heads)


    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._adaLN_zero_path:
            gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaLN(e).chunk(chunks=6, dim=1)
            x = x + alpha_1.unsqueeze(1) * self.attn(self.norm_pre_sa(x) * (1 + gamma_1.unsqueeze(1)) + beta_1.unsqueeze(1))

            if self._cross_attention_context and c is not None:
                x = x + self.cattn_context_module(self.cattn_context_norm(x), c)

            x = x + alpha_2.unsqueeze(1) * self.mlp(self.norm_pre_mlp(x) * (1 + gamma_2.unsqueeze(1)) + beta_2.unsqueeze(1))
        else:
            gamma_1, beta_1, gamma_2, beta_2 = self.film(e).chunk(chunks=4, dim=1)
            x = x + self.attn(self.norm_pre_sa(x) * (1 + gamma_1.unsqueeze(1)) + beta_1.unsqueeze(1))

            if self._cross_attention_context and c is not None:
                x = x + self.cattn_context_module(self.cattn_context_norm(x), c)

            x = x + self.mlp(self.norm_pre_mlp(x) * (1 + gamma_2.unsqueeze(1)) + beta_2.unsqueeze(1))
        return x

class FinalBlock(torch.nn.Module):

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
        self.film = torch.nn.Sequential(
            torch.nn.SiLU(),
            utils.zero_module(torch.nn.Linear(self._embedding_size, 2 * self._input_dims))
        )

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.film(e).chunk(chunks=2, dim=1)
        x = self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.fc1(x)
        return x

class DiT(torch.nn.Module):

    def __init__(
        self,
        input_dims: IntParams,
        hidden_dims: int = 1152,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        depth: int = 12,
        output_channels: Optional[int] = None,
        time_embedding_size: int = 64,
        embedding_size: Optional[int] = None,
        embedder: Optional[torch.nn.Module] = None,
        learnable_embeddings: bool = False,
        learnable_input_embeddings: bool = False,
        learnable_condition_embeddings: bool = False,
        adaLN_zero_path: bool = True,
        cross_attention_condition: bool = False,
        additive_condition: bool = False,
        in_context_condition: bool = False,
        condition_input_dims: Optional[IntParams] = None,
        condition_patch_size: Optional[int] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._patch_size = patch_size
        self._output_channels = self._input_dims[0] if output_channels is None else output_channels
        self._mlp_ratio = mlp_ratio
        self._adaLN_zero_path = adaLN_zero_path

        self._condition_input_dims = condition_input_dims if condition_input_dims is not None else copy.deepcopy(self._input_dims)
        self._condition_patch_size = condition_patch_size if condition_patch_size is not None else self._patch_size
        self._in_context_condition = in_context_condition
        self._cross_attention_condition = cross_attention_condition
        self._additive_condition = additive_condition
        assert  self._in_context_condition ^ self._cross_attention_condition ^ self._additive_condition, (
            "Multiple types of conditioning were selected, but only one type should be"
        )

        self._n_classes = n_classes
        self._has_condition = self._in_context_condition or self._additive_condition or self._cross_attention_condition or (self._n_classes is not None)

        self._time_embedding_size = time_embedding_size
        embedder = FourierEmbeddings if embedder is None else embedder
        self.time_embedder = embedder(d_model=self._time_embedding_size, learnable=learnable_embeddings)
        self.embedding_projection = MLPBlock(
                input_dims=self._time_embedding_size,
                output_dims=self._hidden_dims,
                hidden_dims=[self._hidden_dims],
                activation_function=[torch.nn.SiLU()],
                output_activation=False,
                bias=[True]
        )

        assert not ((self._input_dims[1] % self._patch_size) or (self._input_dims[2] % self._patch_size)), (
            "Input dimensions are not divisable by the patch size!"
        )

        self._grid_h = self._input_dims[1] // self._patch_size
        self._grid_w = self._input_dims[2] // self._patch_size
        self._num_of_patches = self._grid_h * self._grid_w

        self._2d_embeddings = torch.nn.Parameter(
            self._get_2d_positional_embeddings(
                d_model=self._hidden_dims,
                H=self._grid_h,
                W=self._grid_w
            ).float().unsqueeze(0),
            requires_grad=learnable_input_embeddings,
        )
        self.patchify = PatchEmbeddings(
            self._input_dims,
            patch_size=self._patch_size,
            d_model=self._hidden_dims,
            bias=True,
        )

        # TODO: Enable token drop for Classifier-Free Guidance
        if self._has_condition and self._n_classes is not None:
            self.class_embedder = torch.nn.Embedding(self._n_classes, self._hidden_dims)
        else:
            assert not ((self._condition_input_dims[1] % self._condition_patch_size) or (self._condition_input_dims[2] % self._condition_patch_size)), (
                "Condition dimensions are not divisable by the patch size!"
            )
            self._condition_grid_h = self._condition_input_dims[1] // self._condition_patch_size
            self._condition_grid_w = self._condition_input_dims[2] // self._condition_patch_size
            self._num_of_condition_patches = self._condition_grid_h * self._condition_grid_w
            self._2d_embeddings_condition = torch.nn.Parameter(
                self._get_2d_positional_embeddings(
                    d_model=self._hidden_dims,
                    H=self._condition_grid_h,
                    W=self._condition_grid_w,
                ).float().unsqueeze(0),
                requires_grad=learnable_condition_embeddings,
            )

            self.patchify_condition = PatchEmbeddings(
                self._condition_input_dims,
                patch_size=self._condition_patch_size,
                d_model=self._hidden_dims,
                bias=True,
            )

        self.dit_blocks = torch.nn.ModuleList()
        for _ in range(depth):
            self.dit_blocks.append(DiTBlock(
                self._hidden_dims,
                self._mlp_ratio,
                adaLN_zero_path=self._adaLN_zero_path,
                cross_attention_context=self._cross_attention_condition,
            ))

        self.output_layer = FinalBlock(self._hidden_dims, self._output_channels, patch_size=self._patch_size)
        self.initialize_weights()

    def prepare_condition(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        emb = self.embedding_projection(self.time_embedder(t))

        if self._has_condition:
            assert c is not None, "Condition is enabled, but no conditon was provided"

        if self._n_classes is None:
            if self._in_context_condition:
                c = self.patchify_condition(c) + self._2d_embeddings_condition
                x = torch.cat([x, c], dim=1)
            elif self._additive_condition:
                assert self._num_of_patches == self._num_of_condition_patches, (
                    "Additive conditioning requires main input and condition to have the same number of patches."
                )
                c = self.patchify_condition(c) + self._2d_embeddings_condition
                x = x + c
            elif self._cross_attention_condition:
                c = self.patchify_condition(c) + self._2d_embeddings_condition
        else:
            emb = self.class_embedder(c) + emb

        return x, emb, c


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        x = self.patchify(x) + self._2d_embeddings
        x_n_patches = x.size(1)

        x, emb, c = self.prepare_condition(x, t, c)

        for b in self.dit_blocks:
            x = b(x, emb, c if self._cross_attention_condition else None)

        if self._in_context_condition:
            x = x[:, :x_n_patches]

        x = self.output_layer(x, emb)
        x = self._reshape(x)
        return x

    def _get_2d_positional_embeddings(self, d_model: int, W: int, H: int) -> torch.Tensor:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
        grid = torch.meshgrid(yy, xx, indexing='ij')
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

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        # H = W = int(math.sqrt(x.size(1)))
        x = x.view(x.size(0), self._grid_h, self._grid_w, self._patch_size, self._patch_size, self._output_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(x.size(0), self._output_channels, self._patch_size * self._grid_h, self._patch_size * self._grid_w)
        return x
