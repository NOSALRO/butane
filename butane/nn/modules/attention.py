import math
import copy
from typing import TypeAlias, Union, Optional
import torch

from ..._typedefs import *


class SelfAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: Optional[int] = 1,
        dropout_p: Optional[float] = 0.,
        causal: Optional[bool] = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Features cannot be devided equally to N heads"

        self._d_model = d_model
        self._n_heads = n_heads
        self.d_k = self._d_model // self._n_heads
        self._causal = causal

        self.scale_factor = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        self.query = torch.nn.Linear(self._d_model, self._d_model)
        self.key = torch.nn.Linear(self._d_model, self._d_model)
        self.value = torch.nn.Linear(self._d_model, self._d_model)
        self.linear_projection = torch.nn.Linear(self._d_model, self._d_model)

        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert x.dim() == 3, "Attention's input data should be (batch, seq_len, feature_dim)"
        _x_shape = x.shape[:-1]

        _q = self.query(x)
        _k = self.key(x)
        _v = self.value(x)

        if self._n_heads > 1:
            _q = _q.reshape(*_x_shape, self._n_heads, self.d_k).transpose(1,2)
            _k = _k.reshape(*_x_shape, self._n_heads, self.d_k).transpose(1,2)
            _v = _v.reshape(*_x_shape, self._n_heads, self.d_k).transpose(1,2)

        _qk = torch.matmul(_q, _k.transpose(-1, -2)) / self.scale_factor

        if mask is not None:
            _qk.masked_fill_(mask == 0, float('-inf'))

        if self._causal:
            causal_mask = torch.tril(torch.ones(_x_shape[1], _x_shape[1])).to(x.device)
            _qk.masked_fill_(causal_mask == 0, float('-inf'))

        _attention_weights = torch.softmax(_qk, dim=-1)
        _attention_weights = self.dropout(_attention_weights)
        _attention = torch.matmul(_attention_weights, _v)

        if self._n_heads > 1:
            _attention = _attention.transpose(1,2).reshape(*_x_shape, self._d_model)

        return x + self.linear_projection(_attention.contiguous())


