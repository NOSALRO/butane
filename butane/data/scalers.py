from typing import Optional, List, Tuple, Dict, Self, Union
import torch

from .dataset import *


class StandardScaler(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.mu = torch.nn.UninitializedBuffer()
        self.std = torch.nn.UninitializedBuffer()

    def fit(self, X: Union[torch.Tensor, Dataset]) -> None:
        if isinstance(X, Dataset):
            data = X.data_ref()
        else:
            data = X
        self.mu = torch.nn.Buffer(torch.mean(data, (0), keepdim=True))
        self.std = torch.nn.Buffer(torch.std(data, (0), keepdim=True))

    def forward(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x.data
        else:
            data = x
        eps = 1e-08
        data = (data - self.mu) / (self.std + eps)
        if isinstance(x, Dataset):
            x.data = data
        return data

    def reverse(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x.data_ref()
        else:
            data = x
        eps = 1e-08
        data = data * (self.std + eps) + self.mu
        if isinstance(x, Dataset):
            x.data = data
        return data

class MinMaxScaler(torch.nn.Module):

    def __init__(self, min = -1., max = 1.) -> None:
        super().__init__()
        self.x_min = torch.nn.UninitializedBuffer()
        self.x_max = torch.nn.UninitializedBuffer()
        self.min = torch.nn.Buffer(torch.tensor(min, dtype=torch.float32))
        self.max = torch.nn.Buffer(torch.tensor(max, dtype=torch.float32))

    def fit(self, X: Union[torch.Tensor, Dataset]) -> None:
        if isinstance(X, Dataset):
            data = X.data_ref()
        else:
            data = X
        self.x_min = torch.nn.Buffer(torch.min(data, 0, keepdim=True).values)
        self.x_max = torch.nn.Buffer(torch.max(data, 0, keepdim=True).values)

    def forward(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x.data
        else:
            data = x
        x_std = (data - self.x_min) / (self.x_max - self.x_min)
        data = x_std * (self.max - self.min) + self.min
        if isinstance(x, Dataset):
            x.data = data
        return data

    def reverse(self, x: Union[Dataset, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Dataset):
            data = x.data_ref()
        else:
            data = x
        x_std_rev = (data - self.min) / (self.max - self.min)
        data = x_std_rev * (self.x_max - self.x_min) + self.x_min
        if isinstance(x, Dataset):
            x.data = data
        return data
