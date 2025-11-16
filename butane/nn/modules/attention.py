from typing import TypeAlias, Union, Optional, Callable
import torch

from ..._helpers import _fill_defaults, _prod, module_name
from ..utils import utils
from ..._typedefs import *

class _AttentionTemplate(torch.nn.Module):

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

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert x1.dim() == 3, "Attention's input data should be (batch, seq_len, feature_dim)"
        if x2 is None:
            x2 = x1
        assert x2.dim() == 3, "Attention's input data should be (batch, seq_len, feature_dim)"

        _x1_shape = x1.shape[:-1]
        _x2_shape = x2.shape[:-1]

        _q = self.query(x1)
        _k = self.key(x2)
        _v = self.value(x2)

        if self._n_heads > 1:
            _q = _q.reshape(*_x1_shape, self._n_heads, self.d_k).transpose(1, 2)
            _k = _k.reshape(*_x2_shape, self._n_heads, self.d_k).transpose(1, 2)
            _v = _v.reshape(*_x2_shape, self._n_heads, self.d_k).transpose(1, 2)

        _qk = torch.matmul(_q, _k.transpose(-1, -2)) / self.scale_factor

        if mask is not None:
            _qk.masked_fill_(mask == 0, float("-inf"))

        if self._causal:
            causal_mask = torch.tril(torch.ones(_x1_shape[1], _x1_shape[1])).to(x1.device)
            _qk.masked_fill_(causal_mask == 0, float("-inf"))

        _attention_weights = torch.softmax(_qk, dim=-1)
        _attention_weights = self.dropout(_attention_weights)
        _attention = torch.matmul(_attention_weights, _v)

        if self._n_heads > 1:
            _attention = _attention.transpose(1, 2).reshape(*_x1_shape, self._d_model)

        return x1 + self.linear_projection(_attention.contiguous())

class _LocalAttentionTemplate(torch.nn.Module):

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
        zero_conv: bool = False,
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
        self.projection = (
            utils.zero_module(self.conv(self._d_model, self._d_model, 1, bias=bias))
            if zero_conv
            else self.conv(self._d_model, self._d_model, 1, bias=bias)
        )
        # fmt: on

        if dropout_p > 0:
            self.dropout = torch.nn.Dropout(dropout_p)

        self.norm = None
        if prenorm is not None:
            if module_name(prenorm) == "GroupNorm":
                self.norm = prenorm(num_channels=self._d_model)
            elif module_name(prenorm) == "LayerNorm":
                raise ValueError("Cannot use LayerNorm in self-attention")
            else:
                self.norm = prenorm(self._d_model)

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual_ = x1
        if x2 is None:
            x2 = x1
        B1, C1, *spatial1 = x1.shape
        B2, C2, *spatial2 = x2.shape
        print(spatial1)

        if self.N == 1: # When using 1D Attention on images
            x1 = x1.flatten(2)
            x2 = x2.flatten(2)

        L1 = x1.shape[2:].numel()
        L2 = x2.shape[2:].numel()

        if self.norm is not None:
            q_input = self.norm(x1)
            kv_input = self.norm(x2)
        else:
            q_input = x1
            kv_input = x2

        # conv projections
        q = self.query(q_input)
        k = self.key(kv_input)
        v = self.value(kv_input)

        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)

        q = q.transpose(1, 2).reshape(B1, L1, self._n_heads, self.d_k).transpose(1, 2)
        k = k.transpose(1, 2).reshape(B2, L2, self._n_heads, self.d_k).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B2, L2, self._n_heads, self.d_k).transpose(1, 2)

        _attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor

        _attention_weights = torch.softmax(_attention_scores, dim=-1)
        if hasattr(self, "dropout"):
            _attention_weights = self.dropout(_attention_weights)

        _attention = torch.matmul(_attention_weights, v)
        _attention = _attention.transpose(1, 2).reshape(B1, L1, C1).transpose(1, 2)
        _attention = _attention.reshape(B1, C1, *spatial1)

        if self.N == 1:
            _attention = _attention.flatten(2)
            _attention = self.projection(_attention)
            _attention = _attention.reshape(B1, C1, *spatial1)
        else:
            _attention = self.projection(_attention)
        out = residual_ + _attention
        return out

class SelfAttention(_AttentionTemplate):
    def forward(self, x1: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return super().forward(x1, x2=None, mask=mask)

class CrossAttention(_AttentionTemplate): ...

class LocalSelfAttention(_LocalAttentionTemplate):
    def forward(self, x1: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return super().forward(x1, x2=None, mask=mask)

class LocalCrossAttention(_LocalAttentionTemplate): ...

class LocalSelfAttention1d(LocalSelfAttention):
    conv = torch.nn.Conv1d
    N = 1

class LocalSelfAttention2d(LocalSelfAttention):
    conv = torch.nn.Conv2d
    N = 2

class LocalCrossAttention1d(LocalCrossAttention):
    conv = torch.nn.Conv1d
    N = 1

class LocalCrossAttention2d(LocalCrossAttention):
    conv = torch.nn.Conv2d
    N = 2
