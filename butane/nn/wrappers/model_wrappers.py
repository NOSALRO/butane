from typing import Optional, Callable, Tuple
import torch

# class StateModel(torch.nn.Module):

#     def __init__(self, model: torch.nn.Module) -> None:
#         super().__init__()
#         self.model = torch.nn.ModuleList()
#         self.layer_names = []

#         for i in model.modules():
#             self.model.append(i)
#             self.layer_names.append(i.__repr__())

#         self.model = self.model[2:]
#         self.layer_names = self.layer_names[2:]
#         print(self.layer_names)

#     def forward(self, x: torch.Tensor) -> list[torch.Tensor,...]:
#         forward_states = []
#         for i, layer in enumerate(self.model):
#             if not i:
#                 forward_states.append(layer(x))
#             else:
#                 forward_states.append(layer(forward_states[-1]))
#         return forward_states

class TimeDependent(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, condition = None):
        x = torch.hstack([x, t])
        return self.model.forward(x)

