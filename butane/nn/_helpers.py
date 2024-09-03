from typing import Optional, Callable, Union, Tuple, List, Any
import torch

def conv_def(conv_type: str, transpose: Optional[bool] = False) -> Callable[object, object]:
    def inner(cls):
        if conv_type == '1d':
            cls.conv = torch.nn.Conv1d if not transpose else torch.nn.ConvTranspose1d
            cls.pool = torch.nn.MaxPool1d
            cls.norm_type = torch.nn.BatchNorm1d
            cls.N = 1
        elif conv_type == '2d':
            cls.conv = torch.nn.Conv2d if not transpose else torch.nn.ConvTranspose2d
            cls.pool = torch.nn.MaxPool2d
            cls.norm_type = torch.nn.BatchNorm2d
            cls.N = 2
        elif conv_type == '3d':
            cls.conv = torch.nn.Conv3d if not transpose else torch.nn.ConvTranspose3d
            cls.pool = torch.nn.MaxPool3d
            cls.norm_type = torch.nn.BatchNorm3d
            cls.N = 3
        return cls
    return inner

def _fill_defaults(
    vector: Union[List[Any], Tuple[Any,...]],
    size: int,
    size_of_item: Optional[int] = 0
) -> List[Union[int, Tuple[Any,...]]]:

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
    else:
        raise TypeError("Model configure num params error")

def _prod(l: Union[List[Union[int, float]], Tuple[Union[int, float], ...]]) -> Union[float, int]:
    out = 1.
    for i in l:
        out *= i
    return out