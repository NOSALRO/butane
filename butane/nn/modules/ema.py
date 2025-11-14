import torch


class EMA(torch.nn.Module):

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        start_update_at: int = 1,
        update_every: int = 1,
        switch_every: int = -1,
        exclude_bias: bool = False,
    ):
        super().__init__()
        self.model = model
        self.decay = decay
        self._step = 0
        self._start_update_at = start_update_at
        self._update_every = update_every
        self._switch_every = switch_every
        self._exclude_bias = exclude_bias
        self._excluded = set()

        self._param_name_mapping = []
        self._initialize_buffers_from_model(model)

    def _check_exclusion(self, name, param):
        if self._exclude_bias and name.endswith('.bias'):
            return True
        return False

    def _initialize_buffers_from_model(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self._check_exclusion(name, param):
                self._excluded.add(name)
                continue
            buffer_name = name.replace('.', '_')
            if buffer_name not in self._buffers:  # prevents overwriting after load
                self.register_buffer(buffer_name, param.detach().clone())
            self._param_name_mapping.append((buffer_name, param))

    @torch.no_grad()
    def update(self) -> None:
        self._step += 1
        if self._step < self._start_update_at:
            return
        if self._step % self._update_every:
            return
        for buffer_name, param in self._param_name_mapping:
            buffer = getattr(self, buffer_name)
            buffer.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        if not (self._step % self._switch_every) and self._switch_every > 0:
            self.apply()

    def apply(self) -> None:
        # Store the current parameters and apply EMA
        if hasattr(self, "saved_params"):
            raise RuntimeError("EMA.apply() called twice without undo().")
        self.saved_params = {
            name.replace('.', '_'): param.data.clone() 
            for name, param in self.model.named_parameters() 
            if param.requires_grad
        }
        for buffer_name, param in self._param_name_mapping:
            param.data.copy_(getattr(self, buffer_name))

    def undo(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            buffer_name = name.replace('.', '_')
            if buffer_name in self.saved_params:
                param.data.copy_(self.saved_params[buffer_name])
        del self.saved_params

    def load_state_dict(self, state_dict, strict=True):
        self._initialize_buffers_from_model(self.model)
        return super().load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        self.apply()
        model_output = self.model(*args, **kwargs)
        self.undo()
        return model_output
