from typing import Optional, Union, Generator, Tuple, List, Any, Dict, Callable
import math as mth
import torch

def apply_recursively(
    x: Union[torch.Tensor, Tuple, List, Dict[str, torch.Tensor]],
    func: Callable
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]:
    if isinstance(x, (tuple, list)):
        return type(x)(apply_recursively(i, func) for i in x)
    elif isinstance(x, dict):
        return {k: apply_recursively(v, func) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return func(x)
    else:
        return None
    return x

def batching(
    x: Union[torch.Tensor, Tuple[torch.Tensor,...], List[torch.Tensor]],
    batch_size: int = 64,
    *,
    dim: int = 0,
    drop_last: bool = False,
) -> Generator[Union[torch.Tensor, Tuple[torch.Tensor,...], List[torch.Tensor]], None, None]:

    def get_end_step(data: Any, current_end_step: int) -> int:
        if isinstance(data, torch.Tensor):
            size = data.size(dim)
            end_step = size if not drop_last else size - (size % batch_size)
            return min(current_end_step, end_step)
        elif isinstance(data, dict):
            for v in data.values():
                current_end_step = get_end_step(v, current_end_step)
        elif isinstance(data, (list, tuple)):
            for v in data:
                current_end_step = get_end_step(v, current_end_step)
        return current_end_step

    def slice_data(data: Any, start: int, length: int) -> Any:
        if isinstance(data, torch.Tensor):
            return data.narrow(dim=dim, start=start, length=length)
        elif isinstance(data, dict):
            return {k: slice_data(v, start, length) for k, v in data.items()}
        elif isinstance(data, tuple):
            return tuple(slice_data(v, start, length) for v in data)
        elif isinstance(data, list):
            return [slice_data(v, start, length) for v in data]
        return data # Pass through non-tensor data (e.g. None, integers)

    end_step = get_end_step(x, float('inf'))
    if end_step == float('inf'):
        return

    end_step = int(end_step)
    for step in range(0, end_step, batch_size):
        _len = min(batch_size, end_step - step)
        yield slice_data(x, step, _len)

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
