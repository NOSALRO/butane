from typing import Union, Tuple, Callable
import torch


def apply_around_dim(
    operation: Callable,
    x: torch.Tensor, 
    dims: Union[int, Tuple[int]],
    *args,
    **kwargs,
) -> torch.Tensor:

    if isinstance(dims, int):
        dims = [dims]
    all_dims = list(range(x.dim()))
    reduce_dims = tuple(d for d in all_dims if d not in dims)
    try:
        return operation(x, reduce_dims, *args, **kwargs)
    except TypeError:
        reduced_x = x
        for i, d in enumerate(reduce_dims):
            reduced_x = operation(reduced_x, d - i, *args, **kwargs)
            if isinstance(reduced_x, tuple):
                reduced_x = reduced_x.values
        return reduced_x


def sum_around(
    x: torch.Tensor, 
    dims: Union[int, Tuple[int]],
    *,
    keepdim: bool = False
) -> torch.Tensor:

    return apply_around_dim(torch.sum, x, dims, keepdim)

def mean_around(
    x: torch.Tensor, 
    dims: Union[int, Tuple[int]],
    *,
    keepdim: bool = False
) -> torch.Tensor:

    return apply_around_dim(torch.mean, x, dims, keepdim)
