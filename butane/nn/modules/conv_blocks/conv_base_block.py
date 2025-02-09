from typing import List
import torch


class ConvBlockBase(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.input_dims = None

    @property
    def output_size(self) -> torch.Tensor:
        self.eval()
        sz = None
        with torch.no_grad():
            _input = torch.rand(1, *self.input_dims)
            for module in self.children():
                if not isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                    continue
                if isinstance(module, torch.nn.ModuleList):
                    for layer in module:
                        if hasattr(layer, 'input_dims'):
                            _input = torch.rand(1, *layer.input_dims)
                        _input = layer(_input)
                elif isinstance(module, torch.nn.Sequential):
                    _input = module(_input)
            sz = torch.tensor(_input.size()[1:])
        self.train()
        return sz

    @staticmethod
    def _component_output_sz(component: torch.nn.Module, input_dim: List[int]) -> torch.Tensor:
        component.eval()
        with torch.no_grad():
            sz = torch.tensor(component(torch.rand(1, *input_dim)).size())
        component.train()
        return sz

    def _forward_module_list(self, module: torch.nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for layer in module:
            x = layer(x)
        return x

    def _forward_sequential(self, module: torch.nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        x = module(x)
        return x
