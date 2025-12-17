from typing import Optional, Union, Tuple, List, Any
import functools
import torch


def module_name(module: Union[torch.nn.Module, functools.partial]) -> str:
    if module is None:
        return ''
    if isinstance(module, functools.partial):
        return module.func.__name__
    else:
        return module.__name__

def _fill_defaults(
    vector: Union[List[Any], Tuple[Any,...]],
    size: int,
    size_of_item: int = 0
) -> List[Union[int, Tuple[Any,...]]]:

    if not isinstance(vector, (tuple, list)):
        vector = [vector]
    else:
        if len(vector) != size and len(vector) > 1:
            raise ValueError(f"Wrong argument size; Size should be either {size} or 1.")

    filled_vector = []
    if len(vector) == 1:
        for _ in range(size):
            if not isinstance(vector[0], (list, tuple)) and size_of_item != 0:
                filled_vector.append(tuple([vector[0] for _ in range(size_of_item)]))
            else:
                filled_vector.append(vector[0])
        return filled_vector

    elif len(vector) == size and size_of_item != 0:
        for i in range(size):
            if not isinstance(vector[i], (list, tuple)) and size_of_item != 0:
                filled_vector.append(tuple([vector[i] for _ in range(size_of_item)]))
            elif isinstance(vector[i], (list, tuple)) and len(vector[i]) == size_of_item:
                filled_vector.append(vector[i])
        return filled_vector

    elif len(vector) == size and size_of_item == 0:
        return vector

def _prod(l: Union[List[Union[int, float]], Tuple[Union[int, float], ...]]) -> Union[float, int]:
    out = 1.
    for i in l:
        out *= i
    return out
