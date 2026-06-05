from typing import Callable

import torch


class ConditionFusionRegistry(dict):
    def register(self, name: str):
        def wrapper(func: Callable) -> Callable:
            self[name] = func
            return func

        return wrapper


fusion_registry = ConditionFusionRegistry()


class BaseFusion(torch.nn.Module):
    """Abstract base class to handle dynamic Nd broadcasting for flat embeddings."""

    def __init__(self, embedding_dims: int, feature_dims: int, is_projected: bool = False):
        super().__init__()
        self.is_projected = is_projected
        self.feature_dims = feature_dims

    def _broadcast(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims_to_add = target.ndim - context.ndim
        return context[(...,) + (None,) * dims_to_add]


@fusion_registry.register("additive")
class AdditiveFusion(BaseFusion):
    def __init__(self, embedding_dims: int, feature_dims: int):
        super().__init__(embedding_dims, feature_dims)
        self.proj = (
            torch.nn.Linear(embedding_dims, feature_dims)
            if not self.is_projected
            else torch.nn.Identity()
        )

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        bias = self.proj(emb)
        return h + self._broadcast(bias, h)


@fusion_registry.register("film")
class FiLMFusion(BaseFusion):
    def __init__(self, embedding_dims: int, feature_dims: int):
        super().__init__(embedding_dims, feature_dims)
        self.proj = (
            torch.nn.Linear(embedding_dims, feature_dims * 2)
            if not self.is_projected
            else torch.nn.Identity()
        )

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(emb).chunk(2, dim=1)
        gamma = self._broadcast(gamma, h)
        beta = self._broadcast(beta, h)
        return h * (1 + gamma) + beta


@fusion_registry.register("multiplicative")
class MultiplicativeFusion(BaseFusion):
    def __init__(self, embedding_dims: int, feature_dims: int):
        super().__init__(embedding_dims, feature_dims)
        self.proj = (
            torch.nn.Linear(embedding_dims, feature_dims)
            if not self.is_projected
            else torch.nn.Identity()
        )

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale = self.proj(emb)
        return h * self._broadcast(scale, h)


@fusion_registry.register("adagn")
class AdaGNFusion(BaseFusion):
    def __init__(self, embedding_dims: int, feature_dims: int, n_groups: int = 32):
        super().__init__(embedding_dims, feature_dims)
        self.proj = torch.nn.Linear(embedding_dims, feature_dims * 2)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

        # Internal normalization layer
        self.gn = torch.nn.GroupNorm(
            num_groups=n_groups,
            num_channels=feature_dims,
            affine=False,  # Crucial: We supply our own adaptive affine params
        )

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h_norm = self.gn(h)
        gamma, beta = self.proj(emb).chunk(2, dim=1)
        gamma = self._broadcast(gamma, h)
        beta = self._broadcast(beta, h)
        return h_norm * (1 + gamma) + beta
