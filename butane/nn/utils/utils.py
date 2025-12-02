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

def compute_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    grads = [
        p.grad.detach()
        for p in model.parameters()
        if p.grad is not None
    ]

    if not len(grads):
        return 0.0

    device = grads[0].device
    return torch.norm(torch.stack([torch.norm(g, 2.0).to(device) for g in grads]), 2.0)

def calculate_output_size(*modules, input_dims: IntParams) -> torch.Tensor:
    _input = torch.randn(1, *input_dims)
    out_sz = None
    for module in modules:
        if hasattr(module, 'output_size'):
            _input = torch.randn(1, *module.output_size)
            continue
        _input = module(_input)
    return torch.tensor(_input.size())[1:]

@torch.no_grad()
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
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema: ModuleParams = None,
    scaler: Optional[torch.nn.Module] = None
):

    def __load(obj, key):
        if obj is None:
            return None
        if not isinstance(obj, (list, tuple)):
            _state = torch.load(f"{fpath}/{key}.pt", map_location=_device) if cp is None else cp[key]
            obj.load_state_dict(_state)
        elif isinstance(obj, (list, tuple)):
                for i, m in enumerate(obj, start=1):
                    _state = torch.load(f"{fpath}/{key}_{i}.pt", map_location=_device[i]) if cp is None else cp[f'{key}_{i}']
                    m.load_state_dict(_state)
        return obj

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

    cp = None
    if os.path.exists(fpath + "/checkpoint.pt"):
        cp = torch.load(fpath + "/checkpoint.pt", map_location=_device)

    returns["model"] = __load(model, "model")
    returns["optimizer"] = __load(optimizer, "optimizer")
    returns["ema"] = __load(ema, "ema")
    returns["lr_scheduler"] = __load(lr_scheduler, "lr_scheduler")
    returns["scaler"] = __load(scaler, "scaler")

    return returns
