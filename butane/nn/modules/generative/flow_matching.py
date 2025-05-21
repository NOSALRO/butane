from typing import Optional, Callable, Union, Tuple
import time
import functools
import torch
from butane.math import *
# from torchdiffeq import odeint


class FlowMatching(torch.nn.Module):

    def __init__(self, sigma: Optional[float] = 0.1):
        super().__init__()
        # trick to get module's device
        self._dummy_param = torch.nn.Parameter(torch.empty(0))
        self._sigma = sigma

    def sample_timesteps(self, n):
        return torch.rand(n, 1).to(self._dummy_param.device)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @torch.no_grad()
    def flow(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        n_timesteps: int,
        condition: Optional[torch.Tensor] = None,
        keep_record: Optional[bool] = False,
        multiple_gen_per_condition: Optional[bool] = False,
        method: Optional[str] = 'euler',
        reverse: Optional[bool] = False,
        return_model_outputs: Optional[bool] = False,
    ) -> torch.Tensor:

        model.to(self._dummy_param.device)
        if condition is not None:
            condition = condition.to(self._dummy_param.device)

        if not multiple_gen_per_condition:
            x0 = x0.unsqueeze(0)

        if keep_record:
            generated_samples = torch.empty(x0.size(0), n_timesteps, *x0.size()[1:])
            model_outputs = torch.empty(x0.size(0), n_timesteps, *x0.size()[1:])
        else:
            generated_samples = torch.empty_like(x0)
            model_outputs = torch.empty_like(x0)

        if not reverse:
            timesteps = torch.linspace(0., 1., n_timesteps).to(self._dummy_param.device)
        else:
            timesteps = torch.linspace(1., 0., n_timesteps).to(self._dummy_param.device)

        x0 = x0.to(self._dummy_param.device)
        func = lambda t, x, c: model(x, t.repeat(x.size(0)).reshape(-1, 1), c)

        for i, _x0 in enumerate(x0):
            sols, v = odeint(functools.partial(func, c=condition), _x0, timesteps, method, return_model_outputs)
            sols = sols[None,...] if keep_record else sols[-1][None,...]
            generated_samples[i] = sols
            if return_model_outputs:
                v = v[None,...] if keep_record else v[-1][None,...]
                model_outputs[i] = v
        if return_model_outputs:
            return generated_samples.squeeze(0), model_outputs.squeeze(0)
        else:
            return generated_samples.squeeze(0)

    @torch.no_grad()
    def flow_likelihood(
        self,
        model: torch.nn.Module,
        x1: torch.Tensor,
        source_dist,
        n_timesteps: int,
        condition: Optional[torch.Tensor] = None,
        keep_record: Optional[bool] = False,
        multiple_gen_per_condition: Optional[bool] = False,
        method: Optional[str] = 'midpoint',
    ) -> torch.Tensor:

        def func(t, x, c):
            x = x[0]
            with torch.set_grad_enabled(True):
                x.requires_grad_()
                ut = model(x, t.repeat(x.size(0)).unsqueeze(-1), c)
                div = 0
                for i in range(ut.flatten(1).shape[1]):
                    div += torch.autograd.grad(outputs=ut[:,i], inputs=x, grad_outputs=torch.ones_like(ut[:,i]).detach(), create_graph=True)[0][:, i]
                return ut.detach(), div.detach()

        if condition is not None:
            condition = condition.to(self._dummy_param.device)

        if not multiple_gen_per_condition:
            x1 = x1.unsqueeze(0)

        if keep_record:
            generated_samples = torch.empty(x1.size(0), n_timesteps, *x1.size()[1:])
        else:
            generated_samples = torch.empty_like(x1)
        log_likelihoods = torch.empty(*x1.size()[:2])

        timesteps = torch.linspace(1., 0., n_timesteps).to(self._dummy_param.device)
        x1 = x1.to(self._dummy_param.device)

        for i, _x1 in enumerate(x1):
            sols, log_det = odeint(functools.partial(func, c=condition), (_x1, torch.zeros(_x1.size(0), device=_x1.device)), timesteps, method=method)
            x0 = sols[-1].cpu()
            log_p0 = source_dist.log_prob(x0).to(self._dummy_param.device)
            log_det = log_p0 + log_det[-1]

            sols = sols[None,...] if keep_record else sols[-1][None,...]
            log_det = log_det[None,...]
            generated_samples[i] = sols
            log_likelihoods[i] = log_det
        return generated_samples.squeeze(0), log_likelihoods.squeeze(0)

class ConditionalFlowMatching(FlowMatching):

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self._sigma
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon
        u_t = x1 - x0
        return x_t, u_t

class TargetConditionalFlowMatching(FlowMatching):

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = t * x1
        sigma_t = 1 - (1 - self._sigma) * t
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon
        u_t = (x1 - (1 - self._sigma) * x_t) / (1 - (1 - self._sigma) * t)
        return x_t, u_t

class MiddleVarianceFlowMatching(FlowMatching):

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self._sigma * (-4*(t - 0.5)**2 + 1)
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon
        u_t = x1 - x0
        return x_t, u_t
