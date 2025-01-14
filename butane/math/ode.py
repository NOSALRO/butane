import torch


def euler_explicit(model: torch.nn.Module, dt: float, *model_args) -> torch.Tensor:
    v_t = model(*model_args)
    return v_t * dt

def rk4(model: torch.nn.Module, dt: float, *model_args) -> torch.Tensor:
    x, t, sup = model_args[0], model_args[1], model_args[2:]
    k1 = model(x, t, *sup)
    k2 = model(x + (k1 / 2.)*dt, t + dt/2., *sup)
    k3 = model(x + (k2 / 2.)*dt, t + dt/2., *sup)
    k4 = model(x + k3 * dt, t + dt, *sup)
    return (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
