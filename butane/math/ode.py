from typing import Callable, Optional
import functools
import torch

def _euler_explicit(x: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    return x + (dx * dt)

def _rk4(
    func: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    *func_args: Optional[tuple]
) -> torch.Tensor:

    k1 = func(x, t, *func_args)
    k2 = func(x + (k1 / 2.)*dt, t + dt/2., *func_args)
    k3 = func(x + (k2 / 2.)*dt, t + dt/2., *func_args)
    k4 = func(x + k3 * dt, t + dt, *func_args)
    sol = (k1 + 2*k2 + 2*k3 + k4)
    return (dt/6) * sol


def odeint(
    func: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    *func_args: Optional[tuple],
    method: Optional[str] = 'euler_forward'
) -> torch.Tensor:

    time_points = t
    _solutions = torch.full((t.size(0), *x.size()), torch.finfo(torch.float32).max, device=x.device)
    _solutions[0] = x

    for i, (t0, t1) in enumerate(zip(time_points[:-1], time_points[1:])):
        dt = t1 - t0
        if method == 'euler_forward':
            dx = func(_solutions[i], t[i+1], *func_args)
            _solutions[i+1] = _euler_explicit(_solutions[i], dx, dt)
        elif method == 'rk4':
            _solutions[i+1] = _rk4(func, _solutions[i], t[i+1], dt, *func_args)
    return _solutions
