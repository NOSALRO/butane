from typing import TypeAlias, Union, Optional, List, Tuple, Callable
import os
import torch
from ..._typedefs import *
from ..._helpers import module_name


def zero_module(module) -> torch.nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module

def freeze_module(module) -> None:
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module) -> None:
    for p in module.parameters():
        p.requires_grad = True

def calculate_output_size(*modules, input_dims: IntParams) -> torch.Tensor:
    _input = torch.randn(1, *input_dims)
    out_sz = None
    for module in modules:
        if hasattr(module, 'output_size'):
            _input = torch.randn(1, *module.output_size)
            continue
        _input = module(_input)
    return torch.tensor(_input.size())[1:]

@torch.no_grad
def init_weights(model: torch.nn.Module, weight_init_method: Callable, bias_init_method: Optional[Callable] = None):
    for module in model.modules():
        if hasattr(module, 'weight'):
            if 'norm' in module._get_name().lower():
                continue
            if isinstance(module, torch.nn.Embedding):
                continue
            weight_init_method(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None and bias_init_method is not None:
                bias_init_method(module.bias.data)

def load_state(
    fpath: str,
    *,
    model: ModuleParams,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: ModuleParams = None,
    scaler: Optional[torch.nn.Module] = None
):

    returns = {}
    if not os.path.exists(fpath):
        raise("Checkpoint does not exits")

    _device = None
    if isinstance(model, torch.nn.Module):
        _device = next(model.parameters()).device
    elif isinstance(model, (list, tuple)):
        _device = []
        for i, m in enumerate(model):
            _device.append(next(m.parameters()).device)
    else:
        raise ("Model should be either torch.nn.Module or list of torch.nn.Modules")

    if isinstance(model, torch.nn.Module):
        model.load_state_dict(torch.load(f"{fpath}/model.pt", map_location=_device, weights_only=True))
    elif isinstance(model, (list, tuple)):
            for i, m in enumerate(model):
                m.load_state_dict(torch.load(f"{fpath}/model_{i}.pt", map_location=_device[i], weights_only=True))
    returns["model"] = model


    if optimizer is not None:
        optimizer.load_state_dict(torch.load(f"{fpath}/optimizer.pt", map_location=optimizer.param_groups[0]['params'][0].device, weights_only=True))
        returns["optimizer"] = optimizer

    if ema is not None:
        if isinstance(ema, torch.nn.Module):
            ema.load_state_dict(torch.load(f"{fpath}/ema.pt", map_location=_device, weights_only=True))
        elif isinstance(model, (list, tuple)):
                for i, m in enumerate(ema):
                    m.load_state_dict(torch.load(f"{fpath}/ema_{i}.pt", map_location=_device[i], weights_only=True))
        returns["ema"] = ema

    if scaler is not None:
        scaler.load_state_dict(torch.load(f"{fpath}/scaler.pt", map_location=_device, weights_only=True))
        returns["scaler"] = scaler

    return returns
