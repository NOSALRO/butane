from typing import Union, Optional, Callable
import torch


class Transforms:

    def __init__(self, *transformations):
        self._transforms = list(transformations)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        tranformed = x.clone()
        for transform in self._transforms:
            tranformed = transform(tranformed)
        return tranformed

    def __add__(self, t: Callable):
        self._transforms.append(t)
        return self

    def __getitem__(self, idx: int) -> Callable:
        return Transforms(*self._transforms[idx])
