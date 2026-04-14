import copy
import math
import warnings
from functools import partial, reduce
from typing import Any, Type

import torch

from ..._typedefs import *
from ..modules.attention import (
    CrossAttention,
    SelfAttention,
)
from ..modules.embeddings import (
    FourierEmbeddings,
    PatchEmbeddings1d,
    PatchEmbeddings2d,
    PatchEmbeddingsNd,
)
from ..modules.mlp_block import MLPBlock
from ..modules.residual_blocks import *
from ..utils import utils


class DiTBlock(torch.nn.Module):
    _input_dims: int
    _hidden_dims: int
    _embedding_size: int | None
    _cross_attention_context: int
    _cross_attention_heads: int | None
    _adaLN_zero_path: bool

    def __init__(
        self,
        input_dims: int,
        output_ratio: float,
        embedding_size: int | None = None,
        attention_heads: int = 8,
        cross_attention_heads: int | None = None,
        attention_dropout: float = 0.0,
        adaLN_zero_path: bool = True,
        cross_attention_context: bool = False,
    ) -> None:

        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = int(input_dims * output_ratio)
        self._embedding_size = embedding_size if embedding_size is not None else input_dims
        self._cross_attention_context = cross_attention_context
        self._cross_attention_heads = (
            cross_attention_heads if cross_attention_heads is not None else attention_heads
        )
        self._adaLN_zero_path = adaLN_zero_path

        self.norm_pre_sa = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False, eps=1e-06)
        self.norm_pre_mlp = torch.nn.LayerNorm(
            self._input_dims, elementwise_affine=False, eps=1e-06
        )
        self.attn = SelfAttention(
            self._input_dims,
            n_heads=attention_heads,
            dropout_p=attention_dropout,
            apply_residual=False,
        )
        self.mlp = MLPBlock(
            input_dims=self._input_dims,
            output_dims=self._input_dims,
            hidden_dims=[self._hidden_dims],
            activation_function=[torch.nn.GELU(approximate="tanh")],
            output_activation=False,
        )

        if self._adaLN_zero_path:
            self.adaLN = torch.nn.Sequential(
                torch.nn.SiLU(),
                utils.zero_module(torch.nn.Linear(self._embedding_size, 6 * self._input_dims)),
            )
        else:
            self.film = torch.nn.Sequential(
                torch.nn.SiLU(),
                utils.zero_module(torch.nn.Linear(self._embedding_size, 4 * self._input_dims)),
            )

        if self._cross_attention_context:
            self.cattn_context_norm = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False)
            self.cattn_context_module = CrossAttention(
                self._input_dims,
                n_heads=self._cross_attention_heads,
                dropout_p=attention_dropout,
                apply_residual=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._adaLN_zero_path:
            gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaLN(e).chunk(
                chunks=6, dim=1
            )
            x = x + alpha_1.unsqueeze(1) * self.attn(
                self.norm_pre_sa(x) * (1 + gamma_1.unsqueeze(1)) + beta_1.unsqueeze(1)
            )

            if self._cross_attention_context and c is not None:
                x = x + self.cattn_context_module(self.cattn_context_norm(x), c)

            x = x + alpha_2.unsqueeze(1) * self.mlp(
                self.norm_pre_mlp(x) * (1 + gamma_2.unsqueeze(1)) + beta_2.unsqueeze(1)
            )
        else:
            gamma_1, beta_1, gamma_2, beta_2 = self.film(e).chunk(chunks=4, dim=1)
            x = x + self.attn(
                self.norm_pre_sa(x) * (1 + gamma_1.unsqueeze(1)) + beta_1.unsqueeze(1)
            )

            if self._cross_attention_context and c is not None:
                x = x + self.cattn_context_module(self.cattn_context_norm(x), c)

            x = x + self.mlp(
                self.norm_pre_mlp(x) * (1 + gamma_2.unsqueeze(1)) + beta_2.unsqueeze(1)
            )
        return x


class FinalBlock(torch.nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_channels: int,
        patch_size: int = 1,
        embedding_size: int | None = None,
    ) -> None:

        super().__init__()
        self._input_dims = input_dims
        self._output_channels = output_channels
        self._embedding_size = embedding_size if embedding_size is not None else input_dims

        p_size_tuple = (patch_size,) if isinstance(patch_size, int) else patch_size

        self.norm = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False)
        self.fc1 = utils.zero_module(
            torch.nn.Linear(
                self._input_dims, reduce(lambda x, y: x * y, p_size_tuple) * output_channels
            )
        )
        self.film = torch.nn.Sequential(
            torch.nn.SiLU(),
            utils.zero_module(torch.nn.Linear(self._embedding_size, 2 * self._input_dims)),
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


# TODO: Add Multimodal conditioning option.
class DiTNd(torch.nn.Module):
    patch_embedder: type[PatchEmbeddingsNd]
    N: int = -1

    def __init__(
        self,
        input_dims: IntParams,
        hidden_dims: int = 1152,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        depth: int = 12,
        attention_heads: int = 16,
        cross_attention_heads: int | None = None,
        attention_dropout: float = 0.0,
        output_dims: int | None = None,
        time_embedding_size: int = 64,
        time_scaling_coeff: float = 1.0,
        embedding_size: int | None = None,
        embedder: torch.nn.Module | None = None,
        learn_embeddings: bool = False,
        learn_input_embeddings: bool = False,
        learn_condition_embeddings: bool = False,
        adaLN_zero_path: bool = True,
        condition_cross_attention: bool = False,
        condition_additive: bool = False,
        condition_in_context: bool = False,
        condition_input_dims: IntParams | tuple[IntParams, ...] | list[IntParams] | None = None,
        condition_patch_size: IntParams | tuple[IntParams, ...] | list[IntParams] | None = None,
        n_classes: int | None = None,
        cfg_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._patch_size = patch_size
        self._output_dims = self._input_dims[0] if output_dims is None else output_dims
        self._attention_heads = attention_heads
        self._cross_heads = cross_attention_heads
        self._attention_dropout = attention_dropout
        self._mlp_ratio = mlp_ratio
        self._adaLN_zero_path = adaLN_zero_path
        self._learn_embeddings = learn_embeddings
        self._learn_condition_embeddings = learn_condition_embeddings
        self._learn_input_embeddings = learn_input_embeddings

        self._in_context_condition = condition_in_context
        self._cross_attention_condition = condition_cross_attention
        self._additive_condition = condition_additive
        assert (
            self._in_context_condition ^ self._cross_attention_condition ^ self._additive_condition
            or not (
                self._in_context_condition
                or self._cross_attention_condition
                or self._additive_condition
            )
        ), "Multiple types of conditioning were selected, but only one type should be"

        if not isinstance(self._patch_size, (tuple, list)):
            self._patch_size = (self._patch_size,) * (len(self._input_dims) - 1)

        self._n_classes = n_classes
        self._cfg_drop_prob = cfg_drop_prob
        self._has_condition = (
            self._in_context_condition
            or self._additive_condition
            or self._cross_attention_condition
        )

        self._time_scaling_coeff = time_scaling_coeff
        self._time_embedding_size = (
            time_embedding_size if time_embedding_size is not None else self._input_dims[0]
        )
        self._embedding_size = (
            embedding_size if embedding_size is not None else self._time_embedding_size * 4
        )

        embedder = FourierEmbeddings if embedder is None else embedder
        self.time_embedder = embedder(
            d_model=self._embedding_size, learnable=self._learn_embeddings
        )
        self.embedding_projection = MLPBlock(
            input_dims=self._embedding_size,
            output_dims=self._hidden_dims,
            hidden_dims=[self._hidden_dims],
            activation_function=[torch.nn.SiLU()],
            output_activation=False,
            bias=[True],
        )

        assert not reduce(
            lambda x, y: x + y,
            [self._input_dims[1:][i] % self._patch_size[i] for i in range(len(self._patch_size))],
        ), "Input dimensions are not divisable by the patch size!"

        self._input_embeddings = torch.nn.Parameter(
            self._get_positional_embeddings(
                d_model=self._hidden_dims, input_dims=self._input_dims, patch_size=self._patch_size
            )
            .float()
            .unsqueeze(0),
            requires_grad=self._learn_input_embeddings,
        )
        self.patchify = self.patch_embedder(
            self._input_dims,
            patch_size=self._patch_size,
            d_model=self._hidden_dims,
            bias=True,
        )

        # TODO: Enable token drop for Classifier-Free Guidance
        if self._n_classes is not None:
            self.class_embedder = torch.nn.Embedding(
                self._n_classes if self._cfg_drop_prob == 0.0 else self._n_classes + 1,
                self._hidden_dims,
            )
            self._null_class_idx = self._n_classes
        elif self._has_condition:
            self._condition_input_dims = (
                condition_input_dims
                if condition_input_dims is not None
                else copy.deepcopy(self._input_dims)
            )

            self._condition_patch_size = (
                condition_patch_size if condition_patch_size is not None else self._patch_size
            )

            if not isinstance(self._patch_size, (tuple, list)):
                self._patch_size = tuple([self._patch_size] * (len(self._input_dims) - 1))

            if self._is_leaf(self._condition_input_dims):
                if not isinstance(self._condition_patch_size, (tuple, list)):
                    self._condition_patch_size = tuple(
                        [
                            self._condition_patch_size
                            for _ in range(len(self._condition_input_dims) - 1)
                        ]
                    )
            else:
                if isinstance(self._condition_patch_size, (tuple, list)) and not isinstance(
                    self._condition_patch_size, type(self._condition_input_dims)
                ):
                    warnings.warn(
                        f"Ambiguous condition_patch_size {self._condition_patch_size} provided for "
                        f"multimodal input. We will attempt to use this for all modalities, "
                        f"but this may fail if spatial ranks differ."
                    )

            self.patchify_conditions = torch.nn.ModuleDict()
            self.condition_pos_embeds = torch.nn.ParameterDict()
            if self._cfg_drop_prob > 0.0:
                self.null_conditions = torch.nn.ParameterDict()
            self._setup_conditioning_modules(self._condition_input_dims, self._condition_patch_size)

        self.dit_blocks = torch.nn.ModuleList(
            [
                DiTBlock(
                    input_dims=self._hidden_dims,
                    output_ratio=self._mlp_ratio,
                    attention_heads=self._attention_heads,
                    cross_attention_heads=self._cross_heads,
                    attention_dropout=self._attention_dropout,
                    adaLN_zero_path=self._adaLN_zero_path,
                    cross_attention_context=self._cross_attention_condition,
                )
                for _ in range(depth)
            ]
        )

        self.output_layer = FinalBlock(
            self._hidden_dims, self._output_dims, patch_size=self._patch_size
        )
        self.initialize_weights()

    def _prepare_time(self, t: torch.Tensor | None = None) -> torch.Tensor:
        emb = None
        if t is None:
            raise ValueError("Model is time dependent, but time step `t` is None.")
        emb = self.time_embedder(t)
        emb = self.embedding_projection(emb)
        return emb

    def _prepare_labels(
        self,
        emb: torch.Tensor | None,
        y: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self._n_classes is not None:
            if y is None:
                # Classifier-Free Guidance (CFG): Unconditional generation
                # If training with dropout > 0, or inference with no label, we use the NULL token.
                if self._cfg_drop_prob > 0:
                    y = torch.full(
                        (batch_size,), self._null_class_idx, device=device, dtype=torch.long
                    )
                else:
                    raise ValueError(
                        f"Model requires {self._n_classes} class labels, but `y` is None."
                    )
            else:
                assert len(y.shape) == 1
                if self.training and self._cfg_drop_prob > 0:
                    drop_mask = torch.bernoulli(
                        torch.full(y.shape, self._cfg_drop_prob, device=y.device)
                    ).bool()
                    y = torch.where(drop_mask, self._null_class_idx, y)
            y_emb = self.class_embedder(y)
            emb = emb + y_emb if emb is not None else y_emb
        return emb

    def prepare_conditioning(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        c: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,  # Reference
        y: torch.Tensor | None = None,  # Class
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        has_condition_inputs = (c is not None) or (y is not None)
        if (
            (self._has_condition or self._n_classes)
            and not has_condition_inputs
            and self._cfg_drop_prob == 0
        ):
            raise ValueError(
                "Conditioning is enabled but no condition (conditions or labels) was provided."
            )
        elif not (self._has_condition or self._n_classes) and has_condition_inputs:
            raise ValueError("Conditioning is disabled, but conditions were provided.")

        if isinstance(c, dict) and isinstance(self._condition_input_dims, dict):
            missing_keys = [k for k in self._condition_input_dims.keys() if k not in c]
            if missing_keys:
                raise KeyError(
                    f"Condition input is missing required keys defined in 'condition_input_dims'. Missing: {missing_keys}"
                )

        emb = self._prepare_time(t=t)
        emb = self._prepare_labels(emb=emb, y=y, batch_size=x.shape[0], device=x.device)
        c_emb = None

        # Multimodal processing
        if self._has_condition:
            if c is not None:
                processed_conditions = self._encode_conditions(c)
            elif self._cfg_drop_prob > 0.0:
                processed_conditions = [
                    self.null_conditions[k].expand(x.shape[0], -1, -1)
                    for k in sorted(self.null_conditions.keys())
                ]
            if self._in_context_condition or self._cross_attention_condition:
                c_emb = torch.cat(processed_conditions, dim=1) if processed_conditions else None
            elif self._additive_condition:
                c_emb = sum(processed_conditions) if processed_conditions else None

        return (x, emb, c_emb)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor] | None = None,
    ) -> torch.Tensor:

        condition, labels = None, None
        if isinstance(c, dict) and ("condition" in c or "labels" in c):
            condition = c.get("condition")
            labels = c.get("labels")
        else:
            if self._has_condition and self._n_classes is None:
                condition = c
            elif not self._has_condition and self._n_classes is not None:
                labels = c
            else:
                condition = c

        t = t * self._time_scaling_coeff
        t = t.view(-1)
        x, emb, c_emb = self.prepare_conditioning(x, t, c=condition, y=labels)
        return self._forward(x, emb, c_emb)

    def _forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
        c_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.patchify(x) + self._input_embeddings
        x_n_patches = x.size(1)

        if self._in_context_condition and c_emb is not None:
            # Concatenate [x_tokens, condition_tokens]
            x = torch.cat([x, c_emb], dim=1)

        if self._additive_condition and c_emb is not None:
            if x.shape != c_emb.shape:
                raise RuntimeError(
                    f"Additive condition shape {c_emb.shape} does not match "
                    f"input sequence shape {x.shape}. Use cross_attention or in_context instead."
                )
            x = x + c_emb

        for b in self.dit_blocks:
            # If in-context, c_emb is None here. If cross-attention, we pass c_emb.
            b_c_emb = c_emb if self._cross_attention_condition else None
            x = b(x, emb, b_c_emb)

        if self._in_context_condition and c_emb is not None:
            x = x[:, :x_n_patches]

        x = self.output_layer(x, emb)
        x = self._reshape(x)
        return x

    def _encode_conditions(self, c: Any) -> list[torch.Tensor]:
        processed_conditions = []

        def _traverse(data, path):
            if isinstance(data, dict):
                for k in sorted(data.keys()):
                    _traverse(data[k], path + [str(k)])
            elif isinstance(data, (list, tuple)) and not torch.is_tensor(data[0]):
                for i, v in enumerate(data):
                    _traverse(v, path + [str(i)])
            elif torch.is_tensor(data):
                key = ".".join(path) if path else "condition"
                p_cond = self.patchify_conditions[key](data)
                p_cond = p_cond + self.condition_pos_embeds[key]

                if self.training and self._cfg_drop_prob > 0:
                    drop_mask = torch.bernoulli(
                        torch.full(
                            (p_cond.shape[0], 1, 1), self._cfg_drop_prob, device=p_cond.device
                        )
                    ).bool()
                    null_emb = self.null_conditions[key].expand(
                        p_cond.shape[0], p_cond.shape[1], -1
                    )
                    p_cond = torch.where(drop_mask, null_emb, p_cond)

                processed_conditions.append(p_cond)

        _traverse(c, [])
        return processed_conditions

    def _get_positional_embeddings(
        self, d_model: int, input_dims: IntParams, patch_size: int
    ) -> torch.Tensor:
        if len(input_dims) == 3:
            W = input_dims[1] // patch_size[0]
            H = input_dims[2] // patch_size[1]
            return self._get_2d_positional_embeddings(d_model, W=W, H=H)
        elif len(input_dims) == 2:
            return FourierEmbeddings.get_embeddings(
                torch.arange(0, input_dims[-1] // patch_size[0]), d_model
            )

    def _get_2d_positional_embeddings(self, d_model: int, W: int, H: int) -> torch.Tensor:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
        grid = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack(grid, dim=0).unsqueeze(1)
        emb_H = FourierEmbeddings.get_embeddings(grid[0].reshape(-1), d_model // 2)
        emb_W = FourierEmbeddings.get_embeddings(grid[1].reshape(-1), d_model // 2)
        _2d_embeddings = torch.cat([emb_H, emb_W], dim=-1)
        return _2d_embeddings

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
            bias_init_method=partial(torch.nn.init.constant_, val=0),
        )

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        n_patches_per_dim = [d // p for d, p in zip(self._input_dims[1:], self._patch_size)]
        x = x.view(x.size(0), *n_patches_per_dim, *self._patch_size, self._output_dims)
        if self.N == 1:
            x = torch.einsum("nspc->ncsp", x)
        elif self.N == 2:
            x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(
            x.size(0),
            self._output_dims,
            *[d * p for d, p in zip(n_patches_per_dim, self._patch_size)],
        )
        # x = x.reshape(x.size(0), self._output_channels, *self._input_dims[1:])
        return x

    @staticmethod
    def _is_leaf(d):
        """Checks if d is a shape tuple."""
        return isinstance(d, (list, tuple)) and len(d) > 0 and isinstance(d[0], int)

    def _setup_conditioning_modules(self, dims, p_sizes, path=None):
        if path is None:
            if self._is_leaf(dims):
                self._create_modality("condition", dims, p_sizes)
                return
            path = []

        if isinstance(dims, dict):
            for k in sorted(dims.keys()):
                curr_p = p_sizes[k] if isinstance(p_sizes, dict) else p_sizes
                self._setup_conditioning_modules(dims[k], curr_p, path + [str(k)])
        elif isinstance(dims, (list, tuple)) and not self._is_leaf(dims):
            for i, v in enumerate(dims):
                curr_p = p_sizes[i] if isinstance(p_sizes, (list, tuple)) else p_sizes
                self._setup_conditioning_modules(v, curr_p, path + [str(i)])
        else:
            key = ".".join(path)
            self._create_modality(key, dims, p_sizes)

    def _create_modality(self, key, shape, p_size):
        """Helper to actually build the modules for a single modality."""
        # Ensure p_size is broadcasted correctly to the spatial rank
        spatial_rank = len(shape) - 1
        if isinstance(p_size, int):
            p_size = tuple([p_size] * spatial_rank)
        elif isinstance(p_size, (tuple, list)):
            # If a 2D tuple like (8, 8) leaked into a 1D condition, truncate it to (8,)
            # If a 1D tuple like (8,) leaked into a 2D condition, expand it to (8, 8)
            if len(p_size) > spatial_rank:
                p_size = tuple(p_size[:spatial_rank])
            elif len(p_size) < spatial_rank:
                p_size = tuple(list(p_size) + [p_size[-1]] * (spatial_rank - len(p_size)))

        if spatial_rank == 1:
            cond_patch_embedder = PatchEmbeddings1d
        elif spatial_rank == 2:
            cond_patch_embedder = PatchEmbeddings2d

        self.patchify_conditions[key] = cond_patch_embedder(
            shape,
            patch_size=p_size,
            d_model=self._hidden_dims,
            bias=True,
        )

        pos_emb = self._get_positional_embeddings(
            d_model=self._hidden_dims,
            input_dims=shape,
            patch_size=p_size,
        )
        self.condition_pos_embeds[key] = torch.nn.Parameter(
            pos_emb.float().unsqueeze(0),
            requires_grad=self._learn_condition_embeddings,  # Assume you have this flag
        )

        if self._cfg_drop_prob > 0.0:
            self.null_conditions[key] = torch.nn.Parameter(
                torch.randn(1, 1, self._hidden_dims) / math.sqrt(self._hidden_dims),
                requires_grad=True,
            )


class DiT1d(DiTNd):
    patch_embedder = PatchEmbeddings1d
    N = 1


class DiT2d(DiTNd):
    patch_embedder = PatchEmbeddings2d
    N = 2
