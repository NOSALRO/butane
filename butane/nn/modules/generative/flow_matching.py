from typing import Callable, Literal
import time
import functools
import itertools
import torch
# import torchdiffeq
from ....math import *
from ...._utils import *


class FlowMatching(torch.nn.Module):

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        # trick to get module's device
        self.register_buffer("_device_buffer", torch.zeros(1))
        self._sigma = sigma
        self.__source_distribution = None

    def set_source_distribution(self, dist: object):
        self.__source_distribution = dist

    def source_distribution(self) -> object:
        return self.__source_distribution

    def sample_timesteps(self, n: int, skewed: bool = False) -> torch.Tensor:
        # https://github.com/facebookresearch/flow_matching/blob/25ae2d6a672468b58775f47ea086a2a8836be5a4/examples/image/training/train_loop.py#L26
        if skewed:
            mu = -1.2
            std = 1.2
            epsilon = torch.randn((n,), device=self._dummy_param.device)
            sigma = (epsilon * std + mu).exp()
            time = (1 / (1 + sigma)).clamp(0.0001, 1.0)
            return time.unsqueeze(-1)
        else:
            return torch.rand(n, 1, device=self.device)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement forward()")

    @property
    def device(self):
        return self._device_buffer.device

    # Fix condition repettition -> Move it inside the loop, for memory efficiency
    @torch.no_grad()
    def flow(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        n_timesteps: int,
        condition: torch.Tensor | None = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        method: Literal["euler", "heun2", "rk4"] = 'euler',
        reverse: bool = False,
        guidance_scale: float = 1.0,
        return_model_outputs: bool = False,
        edm_time_grid: bool = False,
        batch_size: int = 128,
        target_device: torch.device | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if target_device is None:
            target_device = self.device
        model.to(self.device)
        x0 = x0.to(self.device)
        n_generations = 1
        n_conditions = x0.size(0)
        spatial_dims = x0.shape[1:]

        if condition is not None:
            condition = apply_recursively(condition, lambda x: x.to(self.device))

        if edm_time_grid:
            timesteps = self.edm_time_grid(n_timesteps=n_timesteps, reverse=reverse).to(self.device)
        else:
            timesteps = torch.linspace(
                0.0 if not reverse else 1.0 - 1e-05,
                1.0 - 1e-05 if not reverse else 0.0,
                n_timesteps,
                device=self.device,
            )

        if multiple_gen_per_condition:
            n_generations, n_conditions = x0.size(0), x0.size(1)
            spatial_dims = x0.shape[2:]
            x0 = x0.transpose(0, 1).flatten(0, 1)
            if condition is not None:
                condition = apply_recursively(condition, lambda x: x.repeat_interleave(repeats=n_generations, dim=0))

        def func(t, x, c):
            t = t.expand(x.size(0)).unsqueeze(-1)  # (B, 1)
            if guidance_scale != 1.0 and c is not None:
                v_cond = model(x, t, c)
                v_uncond = model(x, t, None)
                return v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                return model(x, t, c)

        x0_iter = batching(x0, batch_size, dim=0)
        if condition is not None:
            condition_iter = batching(condition, batch_size, dim=0)
        else:
            condition_iter = itertools.repeat(None)

        out_shape = (n_timesteps, x0.size(0), *spatial_dims) if keep_record else (x0.size(0), *spatial_dims)
        xs = torch.empty(out_shape, device=target_device, dtype=x0.dtype)
        vs = torch.empty(out_shape, device=target_device, dtype=x0.dtype) if return_model_outputs else None

        current_idx = 0
        for x0_batch, cond_batch in zip(x0_iter, condition_iter):
            batch_n = x0_batch.size(0)
            x, v = odeint(
                func=functools.partial(func, c=cond_batch),
                x0=x0_batch,
                steps=timesteps,
                method=method,
                return_trajectory=keep_record,
                return_func_outputs=return_model_outputs
            )
            # x = torchdiffeq.odeint(functools.partial(func, c=cond_batch), x0_batch, timesteps, method='explicit_adams')
            # v = torch.zeros_like(x)
            if keep_record:
                xs[:, current_idx: current_idx + batch_n] = x[1:].to(target_device)
                if return_model_outputs:
                    vs[:, current_idx: current_idx + batch_n] = v[1:].to(target_device)
            else:
                xs[current_idx: current_idx + batch_n] = x
                if return_model_outputs:
                    vs[current_idx: current_idx + batch_n] = v
            current_idx += batch_n

        def _revert_shape(x: torch.Tensor):
            return x.view(
                n_timesteps if keep_record else 1,
                n_conditions,
                n_generations if multiple_gen_per_condition else 1,
                *spatial_dims,
            ).movedim((0, 1, 2), (1, 2, 0)).squeeze(1)

        xs = _revert_shape(xs)
        if return_model_outputs:
            vs = _revert_shape(vs)

        if not multiple_gen_per_condition:
            xs = xs.squeeze(0)
            if return_model_outputs: vs = vs.squeeze(0)

        return (xs, vs) if return_model_outputs else xs

    @torch.no_grad()
    def flow_likelihood(
        self,
        model: torch.nn.Module,
        x1: torch.Tensor,
        n_timesteps: int,
        condition: torch.Tensor | None = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        edm_time_grid: bool = False,
        method: Literal["euler", "heun2", "rk4"] = 'euler',
        batch_size: int = 128,
        target_device: torch.device | None = None
    ) -> torch.Tensor:

        if target_device is None:
            target_devcie = self.device
        model.to(self.device)
        x1 = x1.to(self.device)
        n_generations = 1
        n_conditions = x1.size(0)
        spatial_dims = x1.shape[1:]

        if condition is not None:
            condition = apply_recursively(condition, lambda x: x.to(self.device))

        if multiple_gen_per_condition:
            n_generations, n_conditions = x1.size(0), x1.size(1)
            spatial_dims = x1.shape[2:]
            x1 = x1.transpose(0, 1).flatten(0, 1)
            if condition is not None:
                condition = apply_recursively(condition, lambda x: x.repeat_interleave(repeats=n_generations, dim=0))

        z = (torch.randn_like(x1).to(self.device) < 0) * 2.0 - 1.0
        if edm_time_grid:
            timesteps = self.edm_time_grid(n_timesteps=n_timesteps, reverse=True).to(self.device)
        else:
            timesteps = torch.linspace(1, 0, n_timesteps + 1, device=self.device)

        x1_iter = batching(x1, batch_size, dim=0)
        z_iter  = batching(z,  batch_size, dim=0) # Batch z synchronously

        if condition is not None:
            condition_iter = batching(condition, batch_size, dim=0)
        else:
            condition_iter = itertools.repeat(None)

        out_shape = (n_timesteps, x1.size(0), *spatial_dims) if keep_record else (x1.size(0), *spatial_dims)
        xs = torch.empty(out_shape, device=target_device, dtype=x1.dtype)
        lls = torch.empty(x1.size(0), device=target_device, dtype=x1.dtype)

        current_idx = 0

        for x1_batch, z_batch, cond_batch in zip(x1_iter, z_iter, condition_iter):
            batch_n = x1_batch.size(0)

            def func(t, x, c):
                 x_val, _ = x
                 with torch.set_grad_enabled(True):
                    x_val = x_val.detach().requires_grad_(True)

                    t_in = t.expand(x_val.size(0)).unsqueeze(-1)  # (B, 1)
                    ut = model(x_val, t_in, c)

                    # Hutchinson's Trace Estimator
                    ut_dot_z = torch.einsum("ij,ij->i", ut.flatten(1), z_batch.flatten(1))
                    grad_ut_dot_z = torch.autograd.grad(
                        outputs=ut_dot_z,
                        inputs=x_val,
                        grad_outputs=torch.ones_like(ut_dot_z),
                    )[0]
                    div = torch.einsum("ij,ij->i", grad_ut_dot_z.flatten(1), z_batch.flatten(1))
                    return ut.detach(), div.detach()


            init_state = (x1_batch, torch.zeros(batch_n, device=self.device))

            # Returns: ((x_traj, logdet_traj), dx_traj)
            traj, _ = odeint(
                functools.partial(func, c=cond_batch),
                init_state,
                timesteps,
                method=method,
                return_trajectory=True
            )

            x_traj, logdet_traj = traj

            if keep_record:
                xs[:, current_idx : current_idx + batch_n] = x_traj[1:].to(target_device)
            else:
                xs[current_idx : current_idx + batch_n] = x_traj[-1].to(target_device)


            x0_final = x_traj[-1].to(target_device)
            delta_logp = logdet_traj[-1].to(target_device)

            log_p0 = self.__source_distribution.log_prob(x0_final.cpu()).to(target_device)
            log_p0 = log_p0.flatten().to(target_device)
            total_ll = log_p0 + delta_logp.to(target_device)

            lls[current_idx : current_idx + batch_n] = total_ll

            current_idx += batch_n

        def _revert_shape(x: torch.Tensor, is_ll: bool = False):
            if is_ll:
                return x.view(
                    n_conditions,
                    n_generations if multiple_gen_per_condition else 1
                ).movedim((0, 1), (1, 0))

            return x.view(
                n_timesteps if keep_record else 1,
                n_conditions,
                n_generations if multiple_gen_per_condition else 1,
                *spatial_dims,
            ).movedim((0, 1, 2), (1, 2, 0)).squeeze(1)

        xs = _revert_shape(xs, is_ll=False)
        lls = _revert_shape(lls, is_ll=True)
        if not multiple_gen_per_condition:
            xs = xs.squeeze(0)
            lls = lls.squeeze(0)

        return xs, lls

    @torch.no_grad()
    def log_likelihood(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        n_timesteps: int,
        monte_carlo_estiamtes: int = 5,
        condition: torch.Tensor | None = None,
        multiple_gen_per_condition: bool = False,
        edm_time_grid: bool = False,
        method: Literal["euler", "heun2", "rk4"] = 'euler',
        target_device: torch.device | None = None,
    ) -> torch.Tensor:

        x1 = self.flow(
            model=model,
            x0=x0,
            n_timesteps=n_timesteps,
            condition=condition,
            keep_record=False,
            multiple_gen_per_condition=multiple_gen_per_condition,
            method=method,
            return_model_outputs=False,
            edm_time_grid=edm_time_grid,
            target_device=target_device,
        )

        monte_carlo_lls = []
        for _ in range(monte_carlo_estiamtes):
            _, log_likelihood = self.flow_likelihood(
                model=model,
                x1=x1,
                n_timesteps=n_timesteps,
                condition=condition,
                keep_record=False,
                multiple_gen_per_condition=multiple_gen_per_condition,
                edm_time_grid=edm_time_grid,
                target_device=target_device,
            )
            monte_carlo_lls.append(log_likelihood)
        monte_carlo_lls = torch.stack(monte_carlo_lls)
        log_likelihood_estimate = monte_carlo_lls.mean(0)
        return x1, log_likelihood_estimate

    @staticmethod
    def edm_time_grid(n_timesteps: int, r: int = 7, reverse: bool = False):
        sigma_max = 80.0
        sigma_min = 0.002
        r = 7
        t = torch.arange(0, n_timesteps, dtype=torch.float64) / (n_timesteps - 1)
        timesteps = (sigma_max ** (1/r) + t * (sigma_min ** (1/r) - sigma_max**(1/r))) ** r
        timesteps = (timesteps / (1 + timesteps)).squeeze()
        # timesteps = torch.cat([timesteps, torch.full_like(timesteps[:1], t[0])])
        timesteps = torch.cat([timesteps, torch.full_like(timesteps[:1], 1.0)])
        if not reverse:
            timesteps = 1 - timesteps.clamp(0., 1.)
        return timesteps.float()

class ConditionalFlowMatching(FlowMatching):

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

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
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = t * x1
        sigma_t = 1 - (1 - self._sigma) * t
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon
        u_t = (x1 - (1 - self._sigma) * x_t) / (1 - (1 - self._sigma) * t).clamp(min=1e-8)
        return x_t, u_t

class MiddleVarianceFlowMatching(FlowMatching):

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]

        mu_t = t * x1 + (1 - t) * x0

        scale = (-4 * (t - 0.5)**2 + 1)
        sigma_t = self._sigma * scale
        d_sigma_t = self._sigma * (-4 * (2 * t - 1))

        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon

        u_t = (x1 - x0) + d_sigma_t * epsilon
        return x_t, u_t

class CurvedFlowMatching(FlowMatching):

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = (x1 - x0)*(2*t - t**2) + x0
        sigma_t = self._sigma
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon
        u_t = (x1 - x0)*(2 - 2*t)
        return x_t, u_t

    @staticmethod
    def interpolate(
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = (x1 - x0)*(2*t - t**2) + x0
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma * epsilon
        u_t = (x1 - x0)*(2 - 2*t)
        return x_t, u_t
