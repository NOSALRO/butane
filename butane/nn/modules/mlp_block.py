import math
import copy
from typing import TypeAlias, Union, Optional
import torch
from .._typedefs import *
from .._utils import _fill_defaults

class MLPBlock(torch.nn.Sequential):

    def __init__(
        self,
        input_dims: Optional[int] = None,
        output_dims: Optional[int] = None,
        hidden_dims: Optional[Description] = None,
        activation_function: Optional[ModuleParams] = None,
        output_activation: Optional[bool] = False,
        bias: Optional[BoolParams] = [True],
        dropout: Optional[float] = [0.0],
        *,
        architecture: Optional[Architecture] = None,
    ) -> None:

        assert input_dims is not None and output_dims is not None, "Input or Ouput dims cannot be none!"

        super().__init__()
        self.mlp = torch.nn.Sequential()

        _hidden_dims = copy.deepcopy(hidden_dims)
        _hidden_dims.insert(0, input_dims)
        _hidden_dims.insert(len(hidden_dims) + 1, output_dims)
        n_layers = len(_hidden_dims) - 1
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
        return self.mlp(x)

class ProbabilisticMLPBlock(torch.nn.Sequential):

    def __init__(
        self,
        input_dims: Optional[int] = None,
        output_dims: Optional[int] = None,
        hidden_dims: Optional[Description] = None,
        activation_function: Optional[ModuleParams] = None,
        output_activation: Optional[bool] = False,
        bias: Optional[BoolParams] = [True],
        dropout: Optional[float] = [0.0],
    ) -> None:

        assert input_dims is not None and output_dims is not None, "Input or Ouput dims cannot be none!"

        super().__init__()
        self.mlp = torch.nn.Sequential()

        _hidden_dims = copy.deepcopy(hidden_dims)
        _hidden_dims.insert(0, input_dims)
        _hidden_dims.insert(len(hidden_dims) + 1, output_dims)
        n_layers = len(_hidden_dims) - 1
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

        self.mlp.extend(_architecture)

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(_hidden_dims[-2], _hidden_dims[-1], bias = bias[-1]),
            activation_function[-1])

        self.logvar = torch.nn.Sequential(
            torch.nn.Linear(_hidden_dims[-2], _hidden_dims[-1], bias = bias[-1]),
            activation_function[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        mu = self.mu(out)
        logvar = self.logvar(out)
        return mu, logvar
