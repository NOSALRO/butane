from typing import Optional
import torch

from ..._typedefs import *


class LearnableEmbeddings(torch.nn.Module):

    def __init__(self, d_model: int, max_seq_len: Optional[int] = 1000):
        super().__init__()
        self._embeddings = torch.nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), dtype=torch.int32, device=self._embeddings.weight.device)
        x = x + self._embeddings(pos)
        return x

    @property
    def embeddings(self) -> torch.Tensor:
        return self._embeddings.weight

class SinusoidalEmbeddings(torch.nn.Module):

    def __init__(self, d_model: int, max_seq_len: Optional[int] = 1000):
        super().__init__()
        self.__dummy_param = torch.nn.Parameter(torch.empty(0))
        self._PE = torch.zeros((max_seq_len, d_model))
        pos = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        self._PE[:, 0::2] = torch.sin(pos/div_term)
        self._PE[:, 1::2] = torch.cos(pos/div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self._PE[:seq_len, :].to(self.__dummy_param.device)
        return x

    @property
    def embeddings(self) -> torch.Tensor:
        return self._PE.to(self.__dummy_param.device)

    @property
    def dim(self) -> int:
        return self._PE[0].shape[-1]

class PatchEmbeddings(torch.nn.Module):

    def __init__(
        self,
        input_dims: IntParams,
        d_model: int,
        patch_size: Union[int, IntParams],
        bias: Optional[bool] = False,
    ):
        super().__init__()
        self.__input_dims = input_dims
        self._d_model = d_model

        self._image_size = input_dims[1:]

        if not isinstance(patch_size, (tuple, list)):
            patch_size = [patch_size, patch_size]

        self._patch_size = patch_size
        self.n_patches = (self._image_size[0] * self._image_size[1]) / (self._patch_size[0] * self._patch_size[1])

        self.proj = torch.nn.Conv2d(
            self.__input_dims[0],
            self._d_model,
            kernel_size=self._patch_size,
            stride=self._patch_size,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.proj(x)
        patches = patches.flatten(2).transpose(-1,-2)
        return patches


class RelativePositionEmbeddings(torch.nn.Module):

    def __init__(self, d_model: int, max_seq_len: Optional[int] = 1000):
        super().__init__()
        self._k = torch.nn.Embedding(max_seq_len, d_model)
        self._v = torch.nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self._PE[:seq_len, :].to(self.__dummy_param.device)
        return x

    @property
    def embeddings(self) -> torch.Tensor:
        return self._PE.to(self.__dumy_param.device)
