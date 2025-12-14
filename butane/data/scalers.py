from typing import Optional, List, Tuple, Dict, Union, Callable
from abc import ABC, abstractmethod
import torch

from .transforms import Transforms
from ..math.ops import *


class Scaler(ABC, torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.is_fitted = False
        self._fitted_data_shape = None
        self._dims = (1,)

    @abstractmethod
    def _scale(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _unscale(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def fit(
        self,
        X: torch.Tensor,
        dims: Union[int, Tuple[int]] = (1,),
        transforms: Optional[Callable] = None
    ) -> None: ...

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if not self.is_fitted:
            return x

        self.to(device=x.device)
        if inverse:
            out = self._unscale(x)
        else:
             out = self._scale(x)

        if out.ndim > x.ndim:
            out = out.squeeze(0)
        return out

class StandardScaler(Scaler):

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.tensor(0, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(1, dtype=torch.float32))

    def fit(
        self,
        X: torch.Tensor,
        dims: Union[int, Tuple[int]] = (1,),
        transforms: Optional[Callable] = None
    ) -> None:

        self.dims = dims
        if transforms is not None:
            X = transforms(X)
        self._fitted_data_shape = X.shape
        self.register_buffer("mu", apply_around_dim(torch.mean, X, self.dims, keepdim=True))
        self.register_buffer("std", apply_around_dim(torch.std, X, self.dims, keepdim=True) + 1e-12)
        self.is_fitted = True

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.std

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mu

class MinMaxScaler(Scaler):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("min_val", torch.tensor(min_val, dtype=torch.float32))
        self.register_buffer("max_val", torch.tensor(max_val, dtype=torch.float32))
        self.register_buffer("xmax", torch.empty(0, dtype=torch.float32))
        self.register_buffer("xmin", torch.empty(0, dtype=torch.float32))
        self.dims = (1, )

    def fit(
        self,
        X: torch.Tensor,
        dims: Union[int, Tuple[int]] = 1,
        transforms: Optional[Callable] = None
    ) -> None:

        self.dims = dims
        if transforms:
            X = transforms(X)

        self._fitted_data_shape = X.shape
        self.register_buffer("xmin", apply_around_dim(torch.min, X, dims=self.dims, keepdim=True))
        self.register_buffer("xmax", apply_around_dim(torch.max, X, dims=self.dims, keepdim=True))
        self.is_fitted = True

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        x_std = (x - self.xmin) / (self.xmax - self.xmin + eps)
        return x_std * (self.max_val - self.min_val) + self.min_val

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        x_std = (x - self.min_val) / (self.max_val - self.min_val + eps)
        return x_std * (self.xmax - self.xmin) + self.xmin
