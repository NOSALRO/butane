import torch
from typing import TypeAlias, Union, Optional

IntParams: TypeAlias = Optional[Union[list[int], tuple[int, ...]]]
FloatParams: TypeAlias = Optional[Union[list[float], tuple[float, ...]]]
BoolParams: TypeAlias = Optional[Union[bool, list[bool], tuple[bool, ...]]]
ModuleParams: TypeAlias = Optional[Union[list[torch.nn.Module], tuple[torch.nn.Module, ...]]]
Description: TypeAlias = Union[list[tuple[int, torch.nn.Module]], tuple[tuple[int, torch.nn.Module], ...]]
Architecture: TypeAlias = Union[list[tuple[torch.nn.Module, torch.nn.Module]], tuple[tuple[torch.nn.Module, torch.nn.Module], ...]]