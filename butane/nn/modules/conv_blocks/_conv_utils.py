from typing import Optional, Callable
import torch


def define_Nd_convolution(conv_type: str, transpose: Optional[bool] = False) -> Callable:
    def inner(cls):
        if conv_type == '1d':
            if not transpose:
                cls.conv = torch.nn.Conv1d
                cls.pool = torch.nn.MaxPool1d
            else:
                cls.conv = torch.nn.ConvTranspose1d
            cls.norm_type = torch.nn.BatchNorm1d
            cls.N = 1
        elif conv_type == '2d':
            if not transpose:
                cls.conv = torch.nn.Conv2d
                cls.pool = torch.nn.MaxPool2d
            else:
                cls.conv = torch.nn.ConvTranspose2d
            cls.norm_type = torch.nn.BatchNorm2d
            cls.N = 2
        elif conv_type == '3d':
            if not transpose:
                cls.conv = torch.nn.Conv3d
                cls.pool = torch.nn.MaxPool3d
            else:
                cls.conv = torch.nn.ConvTranspose3d
            cls.norm_type = torch.nn.BatchNorm3d
            cls.N = 3
        return cls
    return inner

