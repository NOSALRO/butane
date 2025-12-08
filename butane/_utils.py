from typing import Optional, Union, Generator, Tuple, List
import math as mth
import torch

def batching(
    x: Union[torch.Tensor, Tuple[torch.Tensor,...], List[torch.Tensor]],
    batch_size: int = 64,
    *,
    dim: int = 0,
    drop_last: bool = False,
) -> Generator[Union[torch.Tensor, Tuple[torch.Tensor,...], List[torch.Tensor]], None, None]:

    if isinstance(x, torch.Tensor):
        end_step = x.size(dim) if not drop_last else x.size(dim) - (x.size(dim) % batch_size)

        for step in range(0, end_step, batch_size):
            yield x.index_select(dim, torch.arange(step, min(step + batch_size, x.size(dim)), device=x.device))
    elif isinstance(x, (tuple, list)):
        end_step = x[0].size(dim) if not drop_last else x[0].size(dim) - (x[0].size(dim) % batch_size)
        for xi in x[1:]:
            end_step = min(xi.size(dim) if not drop_last else xi.size(dim) - (xi.size(dim) % batch_size), end_step)

        for step in range(0, end_step, batch_size):
            out = ([xi.index_select(dim, torch.arange(step, min(step + batch_size, end_step), device=xi.device)) for xi in x])
            yield tuple(out) if isinstance(x, tuple) else out

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

def checkerboard_mask(x: torch.Tensor, tile_size: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    padded_height = mth.ceil(height / tile_size) * tile_size
    padded_width = mth.ceil(width / tile_size) * tile_size
    num_tiles = max(padded_height // tile_size, padded_width // tile_size)

    tile_pattern = torch.block_diag(torch.ones((tile_size, tile_size)), torch.ones((tile_size, tile_size)))
    mask = tile_pattern.repeat(num_tiles//2, num_tiles//2)
    mask = mask[:height, :width]
    return (x * mask).to(x.dtype)

def random_mask(x: torch.Tensor, tile_size: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    padded_height = mth.ceil(height / tile_size) * tile_size
    padded_width = mth.ceil(width / tile_size) * tile_size
    num_tiles = max(padded_height // tile_size, padded_width // tile_size)

    tile_pattern = torch.zeros((num_tiles, num_tiles))
    random_mask = torch.randint(0, 2, size=(num_tiles, num_tiles))
    tile_pattern = tile_pattern + random_mask
    tile_block = torch.ones((tile_size, tile_size))
    mask = torch.kron(tile_pattern, tile_block)
    mask = mask[:height, :width]
    return (x * mask).to(x.dtype)
