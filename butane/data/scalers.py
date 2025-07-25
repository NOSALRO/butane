from typing import Optional, List, Tuple, Dict, Union
import torch

from .datasets.dataset import Dataset
from .transforms import Transforms
from ..math.ops import *


class StandardScaler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.empty(0))
        self.register_buffer("std", torch.empty(0))
        self.dims = (1,)

    def fit(
        self,
        X: Union[torch.Tensor, Dataset],
        dims: Union[int, Tuple[int]] = (1,),
        transforms: Optional[Callable] = None
    ) -> None:
        self.dims = dims
        if isinstance(X, Dataset):
            data = X[:]["data"]
        else:
            data = X
        if transforms is not None:
            data = transforms(data)
        self._fitted_data_shape = data.shape
        self.mu = apply_around_dim(torch.mean, data, self.dims, keepdim=True)
        self.std = apply_around_dim(torch.std, data, self.dims, keepdim=True)

    def forward(
        self, 
        x: Union[Dataset, torch.Tensor],
        *,
        feature_idx: Optional[Union[int, torch.Tensor, List[int], Tuple[int,...]]] = None,
        ) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x[:]["data"]
        else:
            data = x

        self.to(data.device)

        if not self.mu.numel() or not self.std.numel():
            return data

        not_batched = len(self._fitted_data_shape) == (len(data.shape) + 1)

        eps = 1e-08
        if feature_idx is None:
            scaled_data = (data - self.mu) / (self.std + eps)
        else: 
            feautre_idx = torch.tensor(feature_idx, dtype=torch.int32)
            scaled_data = (data -  self.mu.index_select(dim=self.dims, index=feautre_idx)) / (self.std.index_select(dim=self.dims, index=feautre_idx) + eps) 

        if not_batched:
            scaled_data = scaled_data.squeeze(0)
        return scaled_data

    def reverse(
        self, 
        x: Union[Dataset, torch.Tensor],
        *,
        feature_idx: Optional[Union[int, torch.Tensor, List[int], Tuple[int,...]]] = None,
    ) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x[:]["data"]
        else:
            data = x

        self.to(data.device)

        if not self.mu.numel() or not self.std.numel():
            return data

        not_batched = len(self._fitted_data_shape) == (len(data.shape) + 1)


        eps = 1e-08
        if feature_idx is None:
            unscaled_data = data * (self.std + eps) + self.mu
        else: 
            feautre_idx = torch.tensor(feature_idx, dtype=torch.int32, device=data.device)
            unscaled_data = data * (self.std.index_select(dim=self.dims, index=feautre_idx) + eps) + self.mu.index_select(dim=self.dims, index=feautre_idx)
        if not_batched:
            unscaled_data = unscaled_data.squeeze(0)
        return unscaled_data


class MinMaxScaler(torch.nn.Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("x_min", None)
        self.register_buffer("x_max", None)
        self.register_buffer("min", torch.tensor(min_val, dtype=torch.float32))
        self.register_buffer("max", torch.tensor(max_val, dtype=torch.float32))
        self.dims = (1, )

    def fit(
        self,
        X: Union[torch.Tensor, Dataset],
        dims: Union[int, Tuple[int]] = 1,
        transforms: Callable = None
    ) -> None:
        self.dims = dims
        if isinstance(X, Dataset):
            data = X[:]["data"]
        else:
            data = X
        if transforms:
            data = transforms(data)

        self._fitted_data_shape = data.shape
        self.x_min = apply_around_dim(torch.min, data, dims=self.dims, keepdim=True)
        self.x_max = apply_around_dim(torch.max, data, dims=self.dims, keepdim=True)

    def forward(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x[:]["data"]
        else:
            data = x

        self.to(data.device)

        if self.x_min is None or self.x_max is None:
            return data


        not_batched = len(self._fitted_data_shape) == (len(data.shape) + 1)

        eps = 1e-8  # prevent division by zero
        x_std = (data - self.x_min) / (self.x_max - self.x_min + eps)
        scaled_data = x_std * (self.max - self.min) + self.min
        if not_batched:
            scaled_data = scaled_data.squeeze(0)
        return scaled_data

    def reverse(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x[:]["data"]
        else:
            data = x

        self.to(data.device)

        if self.x_min is None or self.x_max is None:
            return data

        not_batched = len(self._fitted_data_shape) == (len(data.shape) + 1)

        eps = 1e-8
        x_std = (data - self.min) / (self.max - self.min + eps)
        unscaled_data = x_std * (self.x_max - self.x_min) + self.x_min
        if not_batched:
            unscaled_data = scaled_data.squeeze(0)
        return unscaled_data
