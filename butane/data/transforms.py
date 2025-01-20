from typing import Union, Optional
import torch


class Transforms:

    def __init__(self, *transformations):
        self._transforms = transformations

    def __call__(self, x):
        for transform in self._transforms:
            x = transform(x)
        return x
