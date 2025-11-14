from typing import Callable, Optional
import functools
import torch

@torch.no_grad()
def _euler_explicit(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
    return_func_outputs: bool = False,
) -> torch.Tensor:

    multi_inputs = True
    if not isinstance(x0, (list, tuple)):
        x0 = (x0,)
        multi_inputs = False


    num_vars = len(x0)
    _xs = [torch.empty(len(steps), *x0[i].shape) for i in range(num_vars)]
    [_xs[i][0].copy_(x0[i]) for i in range(num_vars)]
    _dxs = [torch.zeros(len(steps), *x0[i].shape) for i in range(num_vars)] if return_func_outputs else None

    _x = [x.clone() for x in x0]
    for k, (h0, h1) in enumerate(zip(steps[:-1], steps[1:]), start=1):
        _dx = func(h0, *_x)
        if not isinstance(_dx, (list, tuple)):
            _dx = (_dx,)
        for i in range(num_vars):
            _x[i].add_((h1 - h0) * _dx[i])
            _xs[i][k].copy_(_x[i])
            if return_func_outputs:
                _dxs[i][k].copy_(_dx[i])
    return (_xs, _dxs) if multi_inputs else (_xs[0], _dxs[0] if _dxs is not None else None)

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
