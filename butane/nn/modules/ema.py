from typing import Generator, Dict, Any
import contextlib
import torch


class EMA(torch.nn.Module):

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        start_update_at: int = 1,
        update_every: int = 1,
        exclude_bias: bool = False,
    ):
        super().__init__()
        self.model = model
        self.decay = decay
        self._start_update_at = start_update_at
        self._update_every = update_every
        self._exclude_bias = exclude_bias
        self._excluded = set()

        self._step = 0
        self._shadow = {}
        self._backup = {}
        self._param_shadow_pairs = []

        self._initialize_buffers_from_model()

    def _initialize_buffers_from_model(self) -> None:
        self._param_shadow_pairs = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self._exclude_bias and name.endswith('.bias'):
                    continue
                buffer_name = f"ema_{name.replace('.', '_')}"
                if not hasattr(self, buffer_name):
                    self.register_buffer(buffer_name, param.detach().clone())
                shadow_param = getattr(self, buffer_name)
                self._shadow[name] = shadow_param
                self._param_shadow_pairs.append((param, shadow_param))

    @torch.no_grad()
    def update(self) -> None:
        self._step += 1
        if self._step < self._start_update_at:
            return
        if self._step % self._update_every:
            return
        for param, shadow in self._param_shadow_pairs:
            if shadow.device != param.device:
                shadow.data = shadow.to(param.device)
            shadow.lerp_(param, weight=1.0 - self.decay)

    def switch(self) -> None:
        # Switch EMA: https://arxiv.org/pdf/2402.09240
        for param, shadow in self._param_shadow_pairs:
            param.data.copy_(shadow)

    def apply(self) -> None:
        # Store the current parameters and apply EMA
        if len(self._backup) != 0:
            raise RuntimeError("EMA.apply() called twice without undo().")
        for name, param in self.model.named_parameters():
            if name in self._shadow:
                self._backup[name] = param.detach().clone()
                param.data.copy_(self._shadow[name])

    def undo(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> torch.nn.Module:
        super().load_state_dict(state_dict, strict=strict)
        self._initialize_buffers_from_model()
        return self

    @contextlib.contextmanager
    def average_parameters(self) -> Generator[None, None, None]:
        try:
            self.apply()
            yield
        finally:
            self.undo()
