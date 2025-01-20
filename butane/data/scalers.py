from typing import Optional, List, Tuple, Dict, Self, Union
import torch

from .dataset import *
from .transforms import Transforms


class StandardScaler(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.mu = torch.nn.UninitializedBuffer()
        self.std = torch.nn.UninitializedBuffer()

    def fit(self, X: Union[torch.Tensor, Dataset], transforms: Transforms = None) -> None:
        if isinstance(X, Dataset):
            data = X[:]["data"]
        else:
            data = X[:]
        data = data if transforms is None else transforms(data)
        self.mu = torch.nn.Buffer(torch.mean(data, (0), keepdim=False))
        self.std = torch.nn.Buffer(torch.std(data, (0), keepdim=False))

    def forward(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x[:]["data"]
        else:
            data = x

        if isinstance(self.mu, torch.nn.UninitializedBuffer):
            return data

        eps = 1e-08
        data = (data - self.mu) / (self.std + eps)
        return data

    def reverse(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x[:]["data"]
        else:
            data = x

        if isinstance(self.mu, torch.nn.UninitializedBuffer):
            return data

        eps = 1e-08
        data = data * (self.std + eps) + self.mu
        return data

class MinMaxScaler(torch.nn.Module):

    def __init__(self, min = -1., max = 1.) -> None:
        super().__init__()
        self.x_min = torch.nn.UninitializedBuffer()
        self.x_max = torch.nn.UninitializedBuffer()
        self.min = torch.nn.Buffer(torch.tensor(min, dtype=torch.float32))
        self.max = torch.nn.Buffer(torch.tensor(max, dtype=torch.float32))

    def fit(self, X: Union[torch.Tensor, Dataset], transforms: Transforms = None) -> None:
        if isinstance(X, Dataset):
            data = X[:]["data"]
        else:
            data = X
        data = data if transforms is None else transforms(data)
        self.x_min = torch.nn.Buffer(torch.min(data, 0, keepdim=True).values)
        self.x_max = torch.nn.Buffer(torch.max(data, 0, keepdim=True).values)

    def forward(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x[:]["data"]
        else:
            data = x

        if isinstance(self.x_min, torch.nn.UninitializedBuffer):
            return data

        x_std = (data - self.x_min) / (self.x_max - self.x_min)
        data = x_std * (self.max - self.min) + self.min
        return data

    def reverse(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x[:]["data"]
        else:
            data = x

        if isinstance(self.x_min, torch.nn.UninitializedBuffer):
            return data

        x_std_rev = (data - self.min) / (self.max - self.min)
        data = x_std_rev * (self.x_max - self.x_min) + self.x_min
        return data
