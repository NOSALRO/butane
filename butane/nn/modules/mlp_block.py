from typing import Optional
import math
import copy
import torch

from ..._typedefs import *
from ..._helpers import _fill_defaults


class MLPBlock(torch.nn.Sequential):

    def __init__(
        self,
        input_dims: Optional[int] = None,
        output_dims: Optional[int] = None,
        hidden_dims: Optional[IntParams] = None,
        activation_function: Optional[ModuleParams] = [torch.nn.ReLU()],
        output_activation: Optional[bool] = False,
        bias: Optional[BoolParams] = [True],
        dropout: Optional[float] = [0.0],
        *,
        architecture: Optional[Architecture] = None,
    ) -> None:

        assert input_dims is not None and output_dims is not None, "Input or Ouput dims cannot be none!"

        super().__init__()
        self.mlp = torch.nn.ModuleList()

        _hidden_dims = copy.deepcopy(hidden_dims)
        _hidden_dims.insert(0, input_dims)
        _hidden_dims.insert(len(hidden_dims) + 1, output_dims)
        n_layers = len(_hidden_dims) - 1
        activation_function = _fill_defaults(activation_function, n_layers)
        bias = _fill_defaults(bias, n_layers)
        dropout = _fill_defaults(dropout, n_layers)

        if not output_activation:
            activation_function[-1] = torch.nn.Identity()

        # If whole architecture is given initialize it
        if architecture is not None:
            self.mlp.extend(architecture)
        # Construct architecture based on the description provided.
        else:
            _architecture = []
            for i in range(len(_hidden_dims) - 1):
                _architecture.append(torch.nn.Linear(_hidden_dims[i], _hidden_dims[i+1], bias=bias[i]))
                _architecture.append(activation_function[i])
                if dropout[i] != 0.:
                    _architecture.append(torch.nn.Dropout(dropout[i]))

            self.mlp.extend(_architecture)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.mlp, torch.nn.ModuleList):
            for layer in self.mlp:
                x = layer(x)
        elif isinstance(self.mlp, torch.nn.Sequential):
            x = self.mlp(x)
        return x

    def sequential(self):
        self.mlp = torch.nn.Sequential(*self.mlp)
        return self

class ProbabilisticMLPBlock(torch.nn.Sequential):

    def __init__(
        self,
        input_dims: int = None,
        output_dims: int = None,
        hidden_dims: IntParams = None,
        activation_function: Optional[ModuleParams] = [torch.nn.ReLU()],
        output_activation: Optional[bool] = False,
        bias: Optional[BoolParams] = [True],
        dropout: Optional[float] = [0.0],
    ) -> None:

        assert input_dims is not None and output_dims is not None, "Input or Ouput dims cannot be none!"

        super().__init__()
        self.probabilistic_mlp = torch.nn.ModuleList()

        _hidden_dims = copy.deepcopy(hidden_dims)
        _hidden_dims.insert(0, input_dims)
        _hidden_dims.insert(len(hidden_dims) + 1, output_dims)
        n_layers = len(_hidden_dims) - 1
        activation_function = _fill_defaults(activation_function, n_layers)
        bias = _fill_defaults(bias, n_layers)
        dropout = _fill_defaults(dropout, n_layers)

        if not output_activation:
            activation_function[-1] = torch.nn.Identity()

        # Construct architecture
        _architecture = []
        for i in range(len(_hidden_dims) - 2):
            _architecture.append(torch.nn.Linear(_hidden_dims[i], _hidden_dims[i+1], bias=bias[i]))
            _architecture.append(activation_function[i])
            if dropout[i] != 0.:
                _architecture.append(torch.nn.Dropout(dropout[i]))

        self.probabilistic_mlp.extend(_architecture)

        self.mu = torch.nn.ModuleList([
            torch.nn.Linear(_hidden_dims[-2], _hidden_dims[-1], bias = bias[-1]),
            activation_function[-1]])

        self.logvar = torch.nn.ModuleList([
            torch.nn.Linear(_hidden_dims[-2], _hidden_dims[-1], bias = bias[-1]),
            activation_function[-1]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.probabilistic_mlp, torch.nn.ModuleList):
            for layer in self.probabilistic_mlp:
                x = layer(x)
            mu, logvar = x, x
            for layer in self.mu:
                mu = layer(mu)

            for layer in self.logvar:
                logvar = layer(logvar)

        elif isinstance(self.probabilistic_mlp, torch.nn.Sequential):
            out = self.probabilistic_mlp(x)
            mu = self.mu(out)
            logvar = self.logvar(out)

        return mu, logvar

    def sequential(self):
        self.probabilistic_mlp = torch.nn.Sequential(*self.probabilistic_mlp)
        self.mu = torch.nn.Sequential(*self.mu)
        self.logvar = torch.nn.Sequential(*self.logvar)
        return self
