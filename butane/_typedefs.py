import torch
from typing import TypeAlias, Union, Optional

IntParams: TypeAlias = Union[list[int], tuple[int, ...]]
FloatParams: TypeAlias = Union[list[float], tuple[float, ...]]
BoolParams: TypeAlias = Union[bool, list[bool], tuple[bool, ...]]
StrParams: TypeAlias = Union[str, list[str], tuple[str, ...]]
ModuleParams: TypeAlias = Union[list[torch.nn.Module], tuple[torch.nn.Module, ...]]
Description: TypeAlias = Union[list[tuple[int, torch.nn.Module]], tuple[tuple[int, torch.nn.Module], ...]]
Architecture: TypeAlias = Union[list[tuple[torch.nn.Module, torch.nn.Module]], tuple[tuple[torch.nn.Module, torch.nn.Module], ...]]

NestedIntParams: TypeAlias = Union[list[list[int]], tuple[tuple[int, ...], ...]]
NestedFloatParams: TypeAlias = Union[list[list[float]], tuple[tuple[float, ...], ...]]
NestedBoolParams: TypeAlias = Union[bool, list[list[bool]], tuple[tuple[bool, ...], ...]]
NestedStrParams: TypeAlias = Union[str, list[list[str]], tuple[tuple[str, ...], ...]]
NestedModuleParams: TypeAlias = Union[list[list[torch.nn.Module]], tuple[tuple[torch.nn.Module, ...], ...]]
