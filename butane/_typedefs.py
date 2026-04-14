import torch
from typing import Union, Optional, List, Tuple

IntParams = list[int] | tuple[int, ...]
FloatParams = list[float] | tuple[float, ...]
BoolParams = bool | list[bool] | tuple[bool, ...]
StrParams = str | list[str] | tuple[str, ...]
ModuleParams =list[torch.nn.Module] | tuple[torch.nn.Module, ...]
Description = list[tuple[int, torch.nn.Module]] | tuple[tuple[int, torch.nn.Module], ...]
Architecture = list[tuple[torch.nn.Module, torch.nn.Module]] | tuple[tuple[torch.nn.Module, torch.nn.Module], ...]

NestedIntParams = list[list[int]] | tuple[tuple[int, ...], ...]
NestedFloatParams = list[list[float]] | tuple[tuple[float, ...], ...]
NestedBoolParams = bool | list[list[bool]] | tuple[tuple[bool, ...], ...]
NestedStrParams = str | list[list[str]] | tuple[tuple[str, ...], ...]
NestedModuleParams = list[list[torch.nn.Module]] | tuple[tuple[torch.nn.Module, ...], ...]
