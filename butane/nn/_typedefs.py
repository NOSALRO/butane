import torch
from typing import TypeAlias, Union, Optional

IntParams: TypeAlias = Optional[Union[list[int], tuple[int, ...]]]
FloatParams: TypeAlias = Optional[Union[list[float], tuple[float, ...]]]
BoolParams: TypeAlias = Optional[Union[bool, list[bool], tuple[bool, ...]]]
StrParams: TypeAlias = Optional[Union[str, list[str], tuple[str, ...]]]
ModuleParams: TypeAlias = Optional[Union[list[torch.nn.Module], tuple[torch.nn.Module, ...]]]
Description: TypeAlias = Union[list[tuple[int, torch.nn.Module]], tuple[tuple[int, torch.nn.Module], ...]]
Architecture: TypeAlias = Union[list[tuple[torch.nn.Module, torch.nn.Module]], tuple[tuple[torch.nn.Module, torch.nn.Module], ...]]

NestedIntParams: TypeAlias = Optional[Union[list[list[int]], tuple[tuple[int, ...], ...]]]
NestedFloatParams: TypeAlias = Optional[Union[list[list[float]], tuple[tuple[float, ...], ...]]]
NestedBoolParams: TypeAlias = Optional[Union[bool, list[list[bool]], tuple[tuple[bool, ...], ...]]]
NestedStrParams: TypeAlias = Optional[Union[str, list[list[str]], tuple[tuple[str, ...], ...]]]
NestedModuleParams: TypeAlias = Optional[Union[list[list[torch.nn.Module]], tuple[tuple[torch.nn.Module, ...], ...]]]
