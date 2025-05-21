import math
import copy
from typing import TypeAlias, Union, Optional, Callable
from ..._helpers import _fill_defaults, _prod, module_name
import torch

from ..._typedefs import *

class SelfAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: Optional[int] = 1,
        dropout_p: Optional[float] = 0.0,
        causal: Optional[bool] = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Features cannot be devided equally to N heads"

        self._d_model = d_model
        self._n_heads = n_heads
        self.d_k = self._d_model // self._n_heads
        self._causal = causal

        self.scale_factor = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # fmt: off
        self.query = torch.nn.Linear(self._d_model, self._d_model)
        self.key = torch.nn.Linear(self._d_model, self._d_model)
        self.value = torch.nn.Linear(self._d_model, self._d_model)
        self.linear_projection = torch.nn.Linear(self._d_model, self._d_model)
        # fmt: on

        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert x.dim() == 3, "Attention's input data should be (batch, seq_len, feature_dim)"
        _x_shape = x.shape[:-1]

        _q = self.query(x)
        _k = self.key(x)
        _v = self.value(x)

        if self._n_heads > 1:
            _q = _q.reshape(*_x_shape, self._n_heads, self.d_k).transpose(1, 2)
            _k = _k.reshape(*_x_shape, self._n_heads, self.d_k).transpose(1, 2)
            _v = _v.reshape(*_x_shape, self._n_heads, self.d_k).transpose(1, 2)

        _qk = torch.matmul(_q, _k.transpose(-1, -2)) / self.scale_factor

        if mask is not None:
            _qk.masked_fill_(mask == 0, float("-inf"))

        if self._causal:
            causal_mask = torch.tril(torch.ones(_x_shape[1], _x_shape[1])).to(x.device)
            _qk.masked_fill_(causal_mask == 0, float("-inf"))

        _attention_weights = torch.softmax(_qk, dim=-1)
        _attention_weights = self.dropout(_attention_weights)
        _attention = torch.matmul(_attention_weights, _v)

        if self._n_heads > 1:
            _attention = _attention.transpose(1, 2).reshape(*_x_shape, self._d_model)

        return x + self.linear_projection(_attention.contiguous())


class LocalSelfAttention(torch.nn.Module):

    conv: torch.nn.Module
    N: int

    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: Optional[int] = 3,
        n_heads: Optional[int] = 1,
        dropout_p: Optional[float] = 0.0,
        bias: Optional[bool] = False,
        prenorm: ModuleParams = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Features cannot be devided equally to N heads"

        self._d_model = d_model
        self._n_heads = n_heads
        self.d_k = int(self._d_model) // self._n_heads
        _padding = kernel_size // 2

        self.scale_factor = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # fmt: off
        self.query = self.conv(self._d_model, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.key = self.conv(self._d_model, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.value = self.conv(self._d_model, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.projection = self.conv(self._d_model, self._d_model, 1, bias=bias)
        # fmt: on

        if dropout_p > 0:
            self.dropout = torch.nn.Dropout(dropout_p)

        self.norm = None
        if prenorm is not None:
            if module_name(prenorm) == "GroupNorm":
                self.norm = prenorm(num_channels=self._d_model)
            elif module_name(norm_type) == "LayerNorm":
                raise ValueError("Cannot use LayerNorm in self-attention")
            else:
                self.norm = prenorm(self._d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        _outermost_dims, _innermost_dims = torch.tensor(x.shape[:self.N]), torch.tensor(x.shape[self.N:])
        x = x.reshape(*_outermost_dims, -1)

        if self.norm is not None:
            x = self.norm(x)

        _q = self.query(x)
        _k = self.key(x)
        _v = self.value(x)

        if self._n_heads > 1:
            # fmt: off
            _q = _q.reshape(_outermost_dims[0], self._n_heads, self.d_k, _innermost_dims.prod()).transpose(-1, -2)
            _k = _k.reshape(_outermost_dims[0], self._n_heads, self.d_k, _innermost_dims.prod()).transpose(-1, -2)
            _v = _v.reshape(_outermost_dims[0], self._n_heads, self.d_k, _innermost_dims.prod()).transpose(-1, -2)
            # fmt: on

        _qk = torch.matmul(_q, _k.transpose(-1, -2)) / self.scale_factor

        _attention_weights = torch.softmax(_qk, dim=-1)
        if hasattr(self, "dropout"):
            _attention_weights = self.dropout(_attention_weights)
        _attention = torch.matmul(_attention_weights, _v)

        if self._n_heads > 1:
            _attention = _attention.transpose(-1, -2).reshape(x.shape)

        x = x + self.projection(_attention.contiguous())
        return x.reshape(*_outermost_dims, *_innermost_dims)


class LocalSelfAttention1d(LocalSelfAttention):
    conv = torch.nn.Conv1d
    N = 2

class LocalSelfAttention2d(LocalSelfAttention):
    conv = torch.nn.Conv2d
    N = 3
