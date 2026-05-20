from typing import Callable, Union
import torch

class Transforms:
    def __init__(self, *transformations, is_sequence: bool = False):
        self._transforms = list(transformations)
        self.is_sequence = is_sequence

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        transformed = x
        for transform in self._transforms:
            transformed = transform(transformed)
        return transformed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self._transforms:
            return x

        # If it's a sequence, we vmap the ENTIRE pipeline at once!
        if self.is_sequence:
            return torch.vmap(self._forward_single)(x)
        else:
            return self._forward_single(x)

    def __add__(self, t: Callable):
        self._transforms.append(t)
        return self

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            return Transforms(*self._transforms[idx], is_sequence=self.is_sequence)
        return self._transforms[idx]
