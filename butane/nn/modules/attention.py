from typing import Union, Optional, Callable
import math
import torch

from ..._helpers import _fill_defaults, _prod, module_name
from ..utils import utils
from ..._typedefs import *

class _AttentionTemplate(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        *,
        kv_input_size: Optional[int] = None,
        apply_residual: bool = True,
        n_heads: int = 1,
        dropout_p: float = 0.0,
        causal: bool = False,
        flash_attention: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Features cannot be devided equally to N heads"

        self._d_model = d_model
        self._n_heads = n_heads
        self.d_k = self._d_model // self._n_heads
        self._causal = causal
        self._apply_residual = apply_residual
        self._kv_input_size = kv_input_size if kv_input_size is not None else self._d_model
        self._flash_attention = flash_attention

        self.scale_factor = math.sqrt(self.d_k)

        # fmt: off
        self.query = torch.nn.Linear(self._d_model, self._d_model)
        self.key = torch.nn.Linear(self._kv_input_size, self._d_model)
        self.value = torch.nn.Linear(self._kv_input_size, self._d_model)
        self.linear_projection = torch.nn.Linear(self._d_model, self._d_model)
        # fmt: on

        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert x1.dim() >= 3, "Attention's input data should be (batch, seq_len, feature_dim)"
        if x2 is None:
            x2 = x1
        assert x2.dim() >= 3, "Attention's input data should be (batch, seq_len, feature_dim)"

        B, L_Q, *_spatial = x1.shape
        _, L_KV, _ = x2.shape

        if x1.dim() > 3: x1 = x1.flatten(2)
        if x2.dim() > 3: x2 = x2.flatten(2)

        _q = self.query(x1)
        _k = self.key(x2)
        _v = self.value(x2)

        if self._n_heads > 1:
            _q = _q.reshape(B, L_Q, self._n_heads, self.d_k).transpose(1, 2)
            _k = _k.reshape(B, L_KV, self._n_heads, self.d_k).transpose(1, 2)
            _v = _v.reshape(B, L_KV, self._n_heads, self.d_k).transpose(1, 2)

        if not self._flash_attention:
            _qk = torch.matmul(_q, _k.transpose(-1, -2)) / self.scale_factor

            if mask is not None:
                _qk.masked_fill_(mask == 0, float("-inf"))

            if self._causal:
                causal_mask = torch.tril(torch.ones(L_Q, L_KV)).to(x1.device)
                _qk.masked_fill_(causal_mask == 0, float("-inf"))

            _attention_weights = torch.softmax(_qk, dim=-1)
            _attention_weights = self.dropout(_attention_weights)
            _attention = torch.matmul(_attention_weights, _v)
        else:
            _attention = torch.nn.functional.scaled_dot_product_attention(
                query=_q,
                key=_k,
                value=_v,
                attn_mask=mask.bool() if mask is not None else None,
                is_causal=self._causal,
                scale=1.0 / self.scale_factor
            )

        if self._n_heads > 1:
            _attention = _attention.transpose(1, 2).reshape(B, L_Q, self._d_model)

        out = self.linear_projection(_attention.contiguous())
        if self._apply_residual:
            out = x1 + out
        return out.reshape(*out.shape[:-1], *_spatial)

class SelfAttention(_AttentionTemplate):
    def forward(self, x1: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return super().forward(x1, x2=None, mask=mask)

class CrossAttention(_AttentionTemplate): ...

class _SpatialAttentionTemplate(torch.nn.Module):

    conv: torch.nn.Module
    N: int

    def __init__(
        self,
        d_model: int,
        *,
        kv_input_size: Optional[int] = None,
        kv_n_dims: int = None,
        kernel_size: Optional[int] = 1,
        n_heads: Optional[int] = 1,
        dropout_p: Optional[float] = 0.0,
        bias: Optional[bool] = False,
        prenorm: ModuleParams = None,
        zero_out: bool = False,
        apply_residual: bool = False,
        flash_attention: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Features cannot be devided equally to N heads"

        self._d_model = d_model
        self._n_heads = n_heads
        self.d_k = int(self._d_model) // self._n_heads
        self._kv_input_size = kv_input_size if kv_input_size is not None else self._d_model
        self._apply_residual = apply_residual
        self._dropout_p = dropout_p
        self._flash_attention = flash_attention
        _padding = kernel_size // 2

        self.scale_factor = math.sqrt(self.d_k)

        self.kv_conv = self.conv
        if kv_n_dims is not None and kv_n_dims != self.N:
            if kv_n_dims == 1: self.kv_conv = torch.nn.Conv1d
            elif kv_n_dims == 2: self.kv_conv = torch.nn.Conv2d
            elif kv_n_dims == 3: self.kv_conv = torch.nn.Conv3d

        # fmt: off
        self.query = self.conv(self._d_model, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.key = self.kv_conv(self._kv_input_size, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.value = self.kv_conv(self._kv_input_size, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.projection = (
            utils.zero_module(self.conv(self._d_model, self._d_model, 1, bias=bias))
            if zero_out
            else self.conv(self._d_model, self._d_model, 1, bias=bias)
        )
        # fmt: on

        self.dropout = torch.nn.Dropout(self._dropout_p)

        self.norm_1, self.norm_2 = None, None
        if prenorm is not None:
            if module_name(prenorm) == "GroupNorm":
                self.norm_1 = prenorm(num_channels=self._d_model)
                self.norm_2 = prenorm(num_channels=self._kv_input_size)
            elif module_name(prenorm) == "LayerNorm":
                raise ValueError("Cannot use LayerNorm in self-attention")
            else:
                self.norm_1 = prenorm(self._d_model)
                self.norm_2 = prenorm(self._kv_input_size)

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

        if self.N == 1 and len(spatial1) > 1: 
            x1 = x1.flatten(2)
        if self.N == 1 and len(spatial2) > 1:
            x2 = x2.flatten(2)

        L1 = x1.shape[2:].numel()
        L2 = x2.shape[2:].numel()

        q_input = self.norm_1(x1) if self.norm_1 is not None else x1
        kv_input = self.norm_2(x2) if self.norm_2 is not None else x2

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

        if not self._flash_attention:

            _qk = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor

            if mask is not None:
                _qk.masked_fill_(mask == 0, float("-inf"))

            _attention_weights = torch.softmax(_qk, dim=-1)
            _attention_weights = self.dropout(_attention_weights)
            _attention = torch.matmul(_attention_weights, v)
        else:
            _attention = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask.bool() if mask is not None else None,
                dropout_p=self._dropout_p if self.training else 0.0, 
                scale=1.0 / self.scale_factor
            )
        _attention = _attention.transpose(1, 2).reshape(B1, L1, C1).transpose(1, 2)

        if self.N == 1:
            out = self.projection(_attention)
            out = out.reshape(B1, C1, *spatial1)
        else:
            _attention = _attention.reshape(B1, C1, *spatial1)
            out = self.projection(_attention)

        if self._apply_residual:
            out = residual_ + out
        return out

class SpatialSelfAttention(_SpatialAttentionTemplate):
    def forward(self, x1: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return super().forward(x1, x2=None, mask=mask)

class SpatialCrossAttention(_SpatialAttentionTemplate): ...

class SpatialSelfAttention1d(SpatialSelfAttention):
    conv = torch.nn.Conv1d
    N = 1

class SpatialSelfAttention2d(SpatialSelfAttention):
    conv = torch.nn.Conv2d
    N = 2

class SpatialCrossAttention1d(SpatialCrossAttention):
    conv = torch.nn.Conv1d
    N = 1

class SpatialCrossAttention2d(SpatialCrossAttention):
    conv = torch.nn.Conv2d
    N = 2
