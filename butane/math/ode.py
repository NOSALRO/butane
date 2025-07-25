from typing import Callable, Optional
import functools
import torch


def _euler_explicit(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
    return_func_outputs: bool = False,
) -> torch.Tensor:
    solutions = torch.empty((len(steps), *x0.shape), device=x0.device, dtype=x0.dtype)
    solutions[0] = x0

    dx = None
    if return_func_outputs:
        dx = torch.zeros((len(steps), *x0.shape), device=x0.device, dtype=x0.dtype)

    _x = x0
    for i, (h0, h1) in enumerate(zip(steps[:-1], steps[1:]), start=1):
        _dh = h1 - h0
        _dx = func(h0, _x)
        _dxdh = _dx * _dh
        _x = _x + _dxdh
        solutions[i] = _x
        if return_func_outputs:
            dx[i] = _dx
    return solutions, dx

def _euler_explicit_liklihood(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
) -> torch.Tensor:
    solutions = torch.empty((len(steps), *x0[0].shape), device=x0[0].device, dtype=x0[0].dtype)
    likelihoods = torch.empty((len(steps), *x0[1].shape), device=x0[1].device, dtype=x0[1].dtype)
    solutions[0] = x0[0]
    likelihoods[0] = x0[1]

    dx = None
    _x = x0[0]
    _l = x0[1]
    for i, (h0, h1) in enumerate(zip(steps[:-1], steps[1:]), start=1):
        _dh = h1 - h0
        _dx, _dl = func(h0, (_x, _l))
        _dxdh = _dx * _dh
        _dldh = _dl * _dh
        _x = _x + _dxdh
        _l = _l + _dldh
        solutions[i] = _x
        likelihoods[i] = _l
    return solutions, likelihoods, dx



def _rk4(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
    return_func_outputs: bool = False,
) -> torch.Tensor:
    solutions = torch.empty((len(steps), *x0.shape), device=x0.device, dtype=x0.dtype)
    solutions[0] = x0

    dx = None
    if return_func_outputs:
        dx = torch.zeros((len(steps), *x0.shape), device=x0.device, dtype=x0.dtype)

    _x = x0
    for i, (h0, h1) in enumerate(zip(steps[:-1], steps[1:]), start=1):
        _dh = h1 - h0

        k1 = func(h0, _x)
        k2 = func(h0 + _dh / 2.0, _x + (k1 / 2.0) * _dh)
        k3 = func(h0 + _dh / 2.0, _x + (k2 / 2.0) * _dh)
        k4 = func(h0 + _dh, _x + k3 * _dh)
        _dx = k1 + 2 * k2 + 2 * k3 + k4
        _dxdh = _dx * (_dh / 6)
        _x = _x + _dxdh

        solutions[i] = _x
        if return_func_outputs:
            dx[i] = _dx
    return solutions, dx


def odeint(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
    method: Optional[str] = 'euler',
    return_func_outputs: Optional[bool] = False,
) -> torch.Tensor:

    if method == "euler":
        return _euler_explicit(func, x0, steps, return_func_outputs)
    if method == "euler_likelihood":
        return _euler_explicit_liklihood(func, x0, steps)
    elif method == "rk4":
        return _rk4(func, x0, steps, return_func_outputs)
    else:
        raise ValueError("Method does not exist")
