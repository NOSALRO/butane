import copy
from typing import Callable

import torch


class Ensemble(torch.nn.Module):
    def __init__(
        self,
        model_builder: Callable[[], torch.nn.Module],
        depth: int,
    ) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.ensemble_depth = depth
        self.ensemble = torch.nn.ModuleList([model_builder() for _ in range(depth)])

        parameters = []
        for m in self.ensemble:
            parameters.append(torch.nn.utils.parameters_to_vector(m.parameters()))

        if len(parameters) > 1:
            for i in range(1, len(parameters)):
                assert not torch.allclose(parameters[0], parameters[i]), (
                    f"Initialization failed! Model 0 and Model {i} have identical weights."
                )
        del parameters
        self.attachments = {}

    def add_attachment(self, name: str, builder: Callable, *args, **kwargs):
        if name not in self.attachments:
            self.attachments[name] = []

        for m in self.ensemble:
            self.attachments[name].append(builder(m, *args, **kwargs))

    def apply_on_attachment(
        self,
        name: str,
        attachment: str,
        builder: Callable,
        *args,
        **kwargs,
    ):
        if attachment not in self.attachments:
            raise KeyError(f"Source attachment '{source_name}' does not exist.")

        if name not in self.attachments:
            self.attachments[name] = []

        for item in self.attachments[attachment]:
            self.attachments[name].append(builder(item, *args, **kwargs))

    def switch(self, idx: int):
        if idx >= self.ensemble_depth:
            raise IndexError(f"Index {idx} is greater than ensembles depth {self.ensemble_depth}.")

        # Move everything to CPU before switching.
        self._offload_to_cpu()

        _attachments = {}
        for k in list(self.attachments.keys()):
            _attachments[k] = self.attachments[k][idx]
            if hasattr(_attachments[k], "to"):
                _attachments[k].to(self.device)
        return self.ensemble[idx].to(self.device), _attachments

    def _offload_to_cpu(self):
        for i in range(self.ensemble_depth):
            self.ensemble[i].to(torch.device("cpu"))
            for k in list(self.attachments.keys()):
                if hasattr(self.attachments[k][i], "to"):
                    self.attachments[k][i].to(torch.device("cpu"))

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)

        # Inject all attachments under a special, hidden key
        state["__ensemble_attachments__"] = {}
        for name, items in self.attachments.items():
            state["__ensemble_attachments__"][name] = [
                item.state_dict() if hasattr(item, "state_dict") else item for item in items
            ]

        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        # Create a shallow copy so we don't mutate the loaded checkpoint globally
        state_dict = state_dict.copy()

        # Intercept and load the attachments FIRST
        if "__ensemble_attachments__" in state_dict:
            # We use .pop() to remove it so PyTorch's strict loading doesn't crash
            attachments_state = state_dict.pop("__ensemble_attachments__")

            for name, states in attachments_state.items():
                if name in self.attachments:
                    for item, state in zip(self.attachments[name], states):
                        if hasattr(item, "load_state_dict"):
                            item.load_state_dict(state)

        # Pass the cleaned state_dict
        return super().load_state_dict(state_dict, strict=strict)

    def set_device(self, device: torch.device | str):
        self.device = torch.device(device) if isinstance(device, str) else device

    def __iter__(self):
        for i in range(self.ensemble_depth):
            yield self.switch(i)

    @property
    def depth(self) -> int:
        return self.ensemble_depth
