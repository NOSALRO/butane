from typing import Optional, Union, Generator, Tuple, List
import math as mth
import torch

def batching(
    x: torch.Tensor,
    batch_size: int = 64,
    *,
    dim: int = 0,
    drop_last: bool = False,
) -> Generator[torch.Tensor, None, None]:

    end_step = x.size(dim) if not drop_last else x.size(dim) - (x.size(dim) % batch_size)

    for step in range(0, end_step, batch_size):
        yield x.index_select(dim, torch.arange(step, min(step + batch_size, x.size(dim)), device=x.device))

def center_mask(
    x: torch.Tensor,
    mask_size: Optional[Union[int, List[int], Tuple[int, int]]] = None
) -> torch.Tensor:

    H, W = x.shape[-2:]
    mask = torch.ones_like(x)
    mH = H // 2
    mW = W // 2
    if mask_size is None:
        lH = mH // 2
        lW = mW // 2
    elif isinstance(mask_size, int):
        lH = lW = mask_size
    elif isinstance(mask_size, (tuple, list)):
        lH, lW = mask_size
    mask[..., mH - lH:mH + lH, mW - lW:mW + lW] = 0.
    return x * mask

def checkerboard_mask(
    x: torch.Tensor,
    tile_size: int
) -> torch.Tensor:
    height, width = x.shape[-2:]
    padded_height = mth.ceil(height / tile_size) * tile_size
    padded_width = mth.ceil(width / tile_size) * tile_size
    num_tiles = max(padded_height // tile_size, padded_width // tile_size)

    tile_pattern = torch.ones((num_tiles, num_tiles))
    tile_pattern[::2, ::2] = 0.

    tile_block = torch.ones((tile_size, tile_size))
    mask = torch.kron(tile_pattern, tile_block)
    mask = mask[:height, :width]

    return x * mask
