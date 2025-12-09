import torch
from typing import Union, Optional, List, Tuple

IntParams = Union[List[int], Tuple[int, ...]]
FloatParams = Union[List[float], Tuple[float, ...]]
BoolParams = Union[bool, List[bool], Tuple[bool, ...]]
StrParams = Union[str, List[str], Tuple[str, ...]]
ModuleParams = Union[List[torch.nn.Module], Tuple[torch.nn.Module, ...]]
Description = Union[List[Tuple[int, torch.nn.Module]], Tuple[Tuple[int, torch.nn.Module], ...]]
Architecture = Union[List[Tuple[torch.nn.Module, torch.nn.Module]], Tuple[Tuple[torch.nn.Module, torch.nn.Module], ...]]

NestedIntParams = Union[List[List[int]], Tuple[Tuple[int, ...], ...]]
NestedFloatParams = Union[List[List[float]], Tuple[Tuple[float, ...], ...]]
NestedBoolParams = Union[bool, List[List[bool]], Tuple[Tuple[bool, ...], ...]]
NestedStrParams = Union[str, List[List[str]], Tuple[Tuple[str, ...], ...]]
NestedModuleParams = Union[List[List[torch.nn.Module]], Tuple[Tuple[torch.nn.Module, ...], ...]]
