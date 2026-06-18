from typing import Callable

import torch

from ...._typedefs import *
from ...modules.embeddings import FourierEmbeddings


def _get_2d_positional_embeddings(
    embedding_callback_fn: Callable[[torch.Tensor, int], torch.Tensor],
    d_model: int,
    W: int,
    H: int,
) -> torch.Tensor:
    xx = torch.arange(0, W)
    yy = torch.arange(0, H)
    grid = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack(grid, dim=0).unsqueeze(1)
    emb_H = embedding_callback_fn(grid[0].reshape(-1), d_model // 2)
    emb_W = embedding_callback_fn(grid[1].reshape(-1), d_model // 2)
    _2d_embeddings = torch.cat([emb_H, emb_W], dim=-1)
    return _2d_embeddings


def _get_3d_positional_embeddings(
    embedding_callback_fn: Callable[[torch.Tensor, int], torch.Tensor],
    d_model: int,
    D: int,
    H: int,
    W: int,
) -> torch.Tensor:
    dd = torch.arange(0, D)
    yy = torch.arange(0, H)
    xx = torch.arange(0, W)
    grid = torch.meshgrid(dd, yy, xx, indexing="ij")

    # Force the first two dimensions to be perfectly even integers
    dim_D = (d_model // 3) // 2 * 2
    dim_H = (d_model // 3) // 2 * 2
    # Let the third dimension absorb the remaining channels (guaranteed to be even)
    dim_W = d_model - dim_D - dim_H

    emb_D = embedding_callback_fn(grid[0].reshape(-1), dim_D)
    emb_H = embedding_callback_fn(grid[1].reshape(-1), dim_H)
    emb_W = embedding_callback_fn(grid[2].reshape(-1), dim_W)
    _3d_embeddings = torch.cat([emb_D, emb_H, emb_W], dim=-1)
    return _3d_embeddings


def get_positional_embeddings(
    embedding_callback_fn: Callable[[torch.Tensor, int], torch.Tensor],
    d_model: int,
    input_dims: IntParams,
    patch_size: int,
) -> torch.Tensor:
    if len(input_dims) == 4:  # 3D case [C, D, H, W]
        D = input_dims[1] // patch_size[0]
        H = input_dims[2] // patch_size[1]
        W = input_dims[3] // patch_size[2]
        return _get_3d_positional_embeddings(embedding_callback_fn, d_model, D=D, H=H, W=W)
    elif len(input_dims) == 3:  # 2D case [C, H, W]
        W = input_dims[1] // patch_size[0]
        H = input_dims[2] // patch_size[1]
        return _get_2d_positional_embeddings(embedding_callback_fn, d_model, W=W, H=H)
    elif len(input_dims) == 2:  # 1D case [C, L]
        return embedding_callback_fn(torch.arange(0, input_dims[-1] // patch_size[0]), d_model)


def unpatchify(
    x: torch.Tensor,
    input_dims: list[int] | tuple[int],
    patch_size: list[int] | int,
    output_dims: int | None = None,
) -> torch.Tensor:

    output_dims = input_dims[0] if output_dims is None else output_dims
    if not isinstance(patch_size, (tuple, list)):
        patch_size = (patch_size,) * (len(input_dims) - 1)

    n_patches_per_dim = [d // p for d, p in zip(input_dims[1:], patch_size)]
    x = x.view(x.size(0), *n_patches_per_dim, *patch_size, output_dims)
    if x.ndim == 4: # 1D case
        x = torch.einsum("nspc->ncsp", x)
    elif x.ndim == 6: # 2D case
        x = torch.einsum("nhwpqc->nchpwq", x)
    elif x.ndim == 8: # 3D case
        x = torch.einsum("ndhwxyzc->ncdxhywz", x)

    return x.reshape(
        x.size(0),
        output_dims,
        *[d * p for d, p in zip(n_patches_per_dim, patch_size)],
    )
