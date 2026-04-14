import functools
import itertools
import math
import warnings
from typing import Callable, Literal

import torch

from ...._utils import *
from ....math.ops import *
from ...functional import *


def _cosine_beta_scheduler(num_timesteps: int, s: float = 0.008, max_beta: float = 0.999):
    indices = torch.arange(num_timesteps, dtype=torch.float64)
    t1 = indices / num_timesteps
    t2 = (indices + 1) / num_timesteps
    alpha_bar = lambda t: torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    betas = (1 - alpha_bar(t2) / alpha_bar(t1)).clip(max=max_beta)
    return betas.to(torch.float64)


class Diffusion(torch.nn.Module):
    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        scheduler: Literal["linear", "cosine", "scaled_linear"] = "linear",
        *,
        variance_type: Literal[
            "fixed_large",
            "fixed_large_log",
            "fixed_small",
            "fixed_small_log",
            "learned",
            "learned_range",
        ] = "fixed_small",
        prediction_type: Literal["epsilon", "x0", "v"] = "epsilon",
        timestep_spacing: Literal["linspace", "leading", "trailing"] = "linspace",
        scale_timesteps: bool = False,
        clip: bool = True,
        clip_range: list[int] | tuple[int, int] = [-1.0, 1.0],
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.clip = clip
        self.clip_range = clip_range
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value

        self.scale_timesteps = (
            (lambda x: x.float())
            if not scale_timesteps
            else lambda x: x.float() * (1000.0 / self.num_timesteps)
        )
        _variance_options = [
            "fixed_large",
            "fixed_large_log",
            "fixed_small",
            "fixed_small_log",
            "learned",
            "learned_range",
        ]
        assert variance_type in _variance_options, f"Options for variance are: {_variance_options}"
        self.variance_type = variance_type

        if scheduler == "linear":
            betas = torch.linspace(
                start=beta_start, end=beta_end, steps=self.num_timesteps, dtype=torch.float64
            )
        elif scheduler == "cosine":
            betas = _cosine_beta_scheduler(self.num_timesteps, s=0.008, max_beta=0.999)
        elif scheduler == "scaled_linear":
            # Specific for latent diffusion models
            betas = torch.linspace(
                start=beta_start**0.5,
                end=beta_end**0.5,
                steps=self.num_timesteps,
                dtype=torch.float64,
            ).pow(2)
        elif scheduler == "sigmoid":
            betas = torch.linspace(-6, 6, num_timesteps)
            betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{scheduler} is not implemented for {self.__class__}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        zero, one = (
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        )

        alphas_cumprod_prev = torch.cat([one, alphas_cumprod[:-1]], dim=-1)
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], zero], dim=-1)
        betas_tilde = betas * ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("alphas_cumprod_next", alphas_cumprod_next.float())
        self.register_buffer("betas_tilde", betas_tilde.float())

        self.timestep_spacing = timestep_spacing  # store before calling _set_timesteps
        self._set_timesteps()

    def q_params(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates parameters for the forward marginal $q(\\mathbf{x}_t \\vert \\mathbf{x}_0)$.

        The forward diffusion process defines a distribution for $\\mathbf{x}_t$ at any
        arbitrary timestep $t$ directly from the clean image $\\mathbf{x}_0$. This
        marginal distribution is a Gaussian $\\mathcal{N}(\\mathbf{x}_t; \\mu_t, \\sigma_t^2 \\mathbf{I})$
        parameterized by the cumulative product of $\\alpha_t$:

        $$ \\bar{\\alpha}_t = \\prod_{i=1}^t \\alpha_i $$

        The mean $\\mu_t$ and variance $\\sigma_t^2$ are expressed as:

        $$ \\mu_t = \\sqrt{\\bar{\\alpha}_t} \\mathbf{x}_0 $$
        $$ \\sigma_t^2 = 1 - \\bar{\\alpha}_t $$

        This property allows for closed-form sampling of noisy data without
        iteratively applying the Markov chain $q(\\mathbf{x}_t \\vert \\mathbf{x}_{t-1})$.
        """
        t = t.view(-1)
        mu = self._expand(self.alphas_cumprod[t], x_0.shape).sqrt() * x_0
        var = self._expand(1.0 - self.alphas_cumprod[t], x_0.shape)
        return mu, var

    def q_posterior_params(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        step_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates parameters for the posterior q(x_{t-1} | x_t, x_0).

        The posterior $q(\\mathbf{x}_{t-1} \\vert \\mathbf{x}_t, \\mathbf{x}_0)$ is a Gaussian
        distribution $\\mathcal{N}(\\mathbf{x}_{t-1}; \\tilde{\\mu}_t, \\tilde{\beta}_t \\mathbf{I})$
        derived via Bayes' rule. The mean $\\tilde{\\mu}_t$ is a linear combination
        of $\\mathbf{x}_0$ and $\\mathbf{x}_t$:

        $$ \\tilde{\\mu}_t(\\mathbf{x}_t, \\mathbf{x}_0) = \\frac{\\sqrt{\\bar{\\alpha}_{t-1}}\\beta_t}{1-\\bar{\\alpha}_t} \\mathbf{x}_0 + \\frac{\\sqrt{\\alpha_t}(1-\\bar{\\alpha}_{t-1})}{1-\\bar{\\alpha}_t} \\mathbf{x}_t $$

        The variance $\\tilde{\\beta}_t$ is defined as:

        $$ \\tilde{\\beta}_t = \\frac{1-\\bar{\\alpha}_{t-1}}{1-\\bar{\\alpha}_t} \\beta_t $$
        """
        t = t.view(-1)

        if step_idx is not None:
            alpha_prod_prev_t = self._expand(
                self.alphas_cumprod_inference_prev[step_idx].expand(x_t.size(0)), x_t.shape
            )
        else:
            alpha_prod_prev_t = self._expand(self.alphas_cumprod_prev[t], x_t.shape)

        alpha_prod_t = self._expand(self.alphas_cumprod[t], x_t.shape)
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        beta_t = 1 - alpha_prod_t / alpha_prod_prev_t

        coef1 = (alpha_prod_prev_t**0.5 * beta_t) / (1 - alpha_prod_t)
        coef2 = (self._expand(current_alpha_t, x_t.shape) ** 0.5 * (1 - alpha_prod_prev_t)) / (
            1 - alpha_prod_t
        )

        mu = coef1 * x_0 + coef2 * x_t
        var = ((1 - alpha_prod_prev_t) / (1 - alpha_prod_t) * beta_t).clamp(min=1e-20)
        logvar = var.log()
        return mu, var, logvar

    def p_params(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        model_output: torch.Tensor,
        step_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        t = t.view(-1)
        predicted_variance = None
        if x_t.shape[1] == model_output.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.chunk(model_output, chunks=2, dim=1)

        if self.prediction_type == "epsilon":
            # Predict denoised sample directly from x_t
            x_0_hat = self._derive_x_0_from_epsilon(x_t, t, model_output)
        elif self.prediction_type == "x0":
            x_0_hat = model_output
        elif self.prediction_type == "v":
            alphas_cumprod_t = self._expand(self.alphas_cumprod[t], x_t.shape)
            x_0_hat = (alphas_cumprod_t**0.5) * x_t - ((1 - alphas_cumprod_t) ** 0.5) * model_output

        # in p_params, after computing x_0_hat:
        if self.thresholding:
            x_0_hat = self._threshold_sample(x_0_hat)
        elif self.clip:
            x_0_hat = x_0_hat.clamp(self.clip_range[0], self.clip_range[1])

        x_t_minus_1, variance, logvar = self.q_posterior_params(x_0_hat, x_t, t, step_idx=step_idx)
        variance = variance.clamp(min=1e-20)

        if self.variance_type == "fixed_small":
            sigma = variance**0.5
        elif self.variance_type == "fixed_small_log":
            sigma = (0.5 * variance.log()).exp()
        elif self.variance_type == "fixed_large":
            sigma = self._expand(self.betas[t], x_t.shape).sqrt()
        elif self.variance_type == "fixed_large_log":
            sigma = (self._expand(self.betas[t], x_t.shape).log() * 0.5).exp()
        elif self.variance_type == "learned" and predicted_variance is not None:
            sigma = predicted_variance**0.5
        elif self.variance_type == "learned_range" and predicted_variance is not None:
            min_log = variance.log()
            max_log = self._expand(self.betas[t], x_t.shape).log()
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log
            sigma = (0.5 * variance).exp()

        return x_t_minus_1, sigma, x_0_hat

    def forward(
        self,
        x_0: torch.Tensor,
        epsilon: torch.Tensor,
        t: torch.Tensor | int,
    ) -> torch.Tensor:
        mu, var = self.q_params(x_0, t)
        return mu + torch.sqrt(var) * epsilon

    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        model_output: torch.Tensor,
        step_idx: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x_t_minus_1, sigma, _ = self.p_params(x_t, t, model_output, step_idx=step_idx)
        z = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        z_mask = (self.prev_timesteps[step_idx] >= 0).float()
        return x_t_minus_1 + sigma * z * z_mask

    def sample_timesteps(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        t = torch.randint(
            0,
            self.num_timesteps,
            size=(n, 1),
            device=self.betas.device,
            dtype=torch.int64,
            generator=generator,
        )
        return t

    def get_v_target(
        self,
        x_0: torch.Tensor,
        epsilon: torch.Tensor,
        t: torch.Tensor | int,
    ) -> torch.Tensor:
        if self.prediction_type == "v":
            t = t.view(-1)
            sqrt_ab = self._expand(self.alphas_cumprod[t] ** 0.5, x_0.shape)
            sqrt_1ab = self._expand((1 - self.alphas_cumprod[t]) ** 0.5, x_0.shape)
            return sqrt_ab * epsilon - sqrt_1ab * x_0
        else:
            raise ValueError("Prediction type is not set to 'v'")

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        condition: torch.Tensor | None = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        return_model_outputs: bool = False,
        guidance_scale: float = 1.0,
        num_inference_steps: int | None = None,
        timestep_spacing: Literal["linspace", "leading", "trailing"] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if num_inference_steps is not None or timestep_spacing is not None:
            self._set_timesteps(num_inference_steps, timestep_spacing)

        result = self._sample_template(
            sample_fn=self.reverse,
            model=model,
            x_T=x_T,
            condition=condition,
            keep_record=keep_record,
            multiple_gen_per_condition=multiple_gen_per_condition,
            reverse=False,
            return_model_outputs=return_model_outputs,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if num_inference_steps is not None or timestep_spacing is not None:
            self._set_timesteps()
        return result

    @torch.no_grad()
    def sample_ddim(
        self,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        condition: torch.Tensor | None = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        reverse: bool = False,
        return_model_outputs: bool = False,
        guidance_scale: float = 1.0,
        num_inference_steps: int | None = None,
        timestep_spacing: Literal["linspace", "leading", "trailing"] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        sample_fn = self._ddim if not reverse else functools.partial(self._ddim, reverse=True)

        # temporarily override timesteps for this call
        if num_inference_steps is not None or timestep_spacing is not None:
            self._set_timesteps(num_inference_steps, timestep_spacing)

        result = self._sample_template(
            sample_fn=sample_fn,
            model=model,
            x_T=x_T,
            condition=condition,
            keep_record=keep_record,
            multiple_gen_per_condition=multiple_gen_per_condition,
            reverse=reverse,
            return_model_outputs=return_model_outputs,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        if num_inference_steps is not None or timestep_spacing is not None:
            self._set_timesteps()
        return result

    @torch.no_grad()
    def inpainting(
        self,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        x_original: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        multiple_gen_per_mask: bool = False,
        keep_record: bool = False,
        return_model_outputs: bool = False,
        resample_steps: int = 1,
        batch_size: int = 128,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        _device = self.betas.device
        model.to(_device)
        x_T = x_T.to(_device)
        x_original = x_original.to(_device)
        mask = mask.to(_device)
        n_generations = 1
        n_conditions = x_T.size(0)
        spatial_dims = x_T.shape[1:]

        if condition is not None:
            condition = apply_recursively(condition, lambda x: x.to(_device))

        if multiple_gen_per_mask:
            n_generations, n_conditions = x_T.size(0), x_T.size(1)
            spatial_dims = x_T.shape[2:]
            x_T = x_T.transpose(0, 1).flatten(0, 1)  # (n_cond * n_gen, *spatial)
            x_original = x_original.repeat_interleave(n_generations, dim=0)
            mask = mask.repeat_interleave(n_generations, dim=0)
            if condition is not None:
                condition = apply_recursively(
                    condition, lambda x: x.repeat_interleave(repeats=n_generations, dim=0)
                )

        timesteps = self.timesteps
        n_timesteps = len(timesteps)

        x_T_iter = batching(x_T, batch_size, dim=0)
        x_orig_iter = batching(x_original, batch_size, dim=0)
        mask_iter = batching(mask, batch_size, dim=0)
        if condition is not None:
            condition_iter = batching(condition, batch_size, dim=0)
        else:
            condition_iter = itertools.repeat(None)

        out_shape = (
            (n_timesteps, x_T.size(0), *spatial_dims)
            if keep_record
            else (x_T.size(0), *spatial_dims)
        )
        xs = torch.empty(out_shape, device=_device, dtype=x_T.dtype)
        vs = (
            torch.empty(out_shape, device=_device, dtype=x_T.dtype)
            if return_model_outputs
            else None
        )

        current_idx = 0
        for x_T_batch, x_orig_batch, mask_batch, cond_batch in zip(
            x_T_iter, x_orig_iter, mask_iter, condition_iter
        ):
            batch_n = x_T_batch.size(0)
            x = x_T_batch

            for j, t in enumerate(timesteps):
                _t = torch.full((batch_n, 1), t, device=_device, dtype=torch.int64)

                for r in range(resample_steps):
                    # forward diffuse x_original to t
                    epsilon_known = torch.randn_like(x_orig_batch, generator=generator)
                    x_known_noisy = self.forward(x_orig_batch, epsilon_known, _t)

                    # denoise with model
                    if guidance_scale != 1.0 and cond_batch is not None:
                        out_cond = model(x, self.scale_timesteps(_t), cond_batch)
                        out_uncond = model(x, self.scale_timesteps(_t), None)
                        out = out_uncond + guidance_scale * (out_cond - out_uncond)
                    else:
                        out = model(x, self.scale_timesteps(_t), cond_batch)

                    x_unknown_denoised = self.reverse(x, _t, out, step_idx=j, generator=generator)

                    # combine
                    x = mask_batch * x_known_noisy + (1 - mask_batch) * x_unknown_denoised

                    prev_t_val = self.prev_timesteps[j].item()
                    if r < resample_steps - 1 and prev_t_val >= 0:
                        _t_prev = torch.full(
                            (batch_n, 1), prev_t_val, device=_device, dtype=torch.int64
                        )
                        epsilon_resample = torch.randn_like(x, generator=generator)
                        mu_t, var_t = self.q_params(x, _t_prev)
                        x = mu_t + torch.sqrt(var_t) * epsilon_resample

                if keep_record:
                    xs[j, current_idx : current_idx + batch_n] = x
                    if return_model_outputs:
                        vs[j, current_idx : current_idx + batch_n] = self._get_epsilon_from_model(
                            x, _t, out
                        )

            if not keep_record:
                xs[current_idx : current_idx + batch_n] = x
                if return_model_outputs:
                    vs[current_idx : current_idx + batch_n] = self._get_epsilon_from_model(
                        x, _t, out
                    )

            current_idx += batch_n

        def _revert_shape(tensor_to_reshape: torch.Tensor):
            return (
                tensor_to_reshape.view(
                    n_timesteps if keep_record else 1,
                    n_conditions,
                    n_generations if multiple_gen_per_mask else 1,
                    *spatial_dims,
                )
                .movedim((0, 1, 2), (1, 2, 0))
                .squeeze(1)
            )

        xs = _revert_shape(xs)
        if return_model_outputs:
            vs = _revert_shape(vs)

        if not multiple_gen_per_mask:
            xs = xs.squeeze(0)
            if return_model_outputs:
                vs = vs.squeeze(0)

        return (xs, vs) if return_model_outputs else xs

    @torch.no_grad()
    def _ddim(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        model_output: torch.Tensor,
        eta: float = 0.0,
        reverse: bool = False,
        step_idx: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        t = t.view(-1)
        epsilon_hat = self._get_epsilon_from_model(x_t, t, model_output)
        _, _, x_0_hat = self.p_params(x_t, t, model_output, step_idx=step_idx)

        alpha_prod_t = self._expand(self.alphas_cumprod[t], x_t.shape)

        if not reverse:
            if step_idx is not None:
                alpha_prod_prev_t = self._expand(
                    self.alphas_cumprod_inference_prev[step_idx].expand(x_t.size(0)), x_t.shape
                )
            else:
                alpha_prod_prev_t = self._expand(self.alphas_cumprod_prev[t], x_t.shape)
            sigma = eta * torch.sqrt(
                (1 - alpha_prod_prev_t)
                / (1 - alpha_prod_t)
                * (1 - alpha_prod_t / alpha_prod_prev_t)
            )
            mu = (
                x_0_hat * torch.sqrt(alpha_prod_prev_t)
                + torch.sqrt(1 - alpha_prod_prev_t - sigma**2) * epsilon_hat
            )
            noise = torch.randn_like(x_t, generator=generator)
            z_mask = (self.prev_timesteps[step_idx] >= 0).float()
            return mu + sigma * noise * z_mask
        else:
            alpha_prod_t_next = (
                self._expand(
                    self.alphas_cumprod_inference_next[step_idx].expand(x_t.size(0)), x_t.shape
                )
                if step_idx is not None
                else self._expand(self.alphas_cumprod_next[t], x_t.shape)
            )
            return (
                torch.sqrt(alpha_prod_t_next) * x_0_hat
                + torch.sqrt(1 - alpha_prod_t_next) * epsilon_hat
            )

    @staticmethod
    def _expand(x, shape: tuple):
        while x.dim() != len(shape):
            x = x[..., None]
        return x

    def _derive_x_0_from_epsilon(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        t = t.view(-1)
        sqrt_alphas_cumprod_t = self._expand(torch.sqrt(self.alphas_cumprod[t]), x_t.shape)
        sqrt_1_minus_alphas_cumprod_t = self._expand(
            torch.sqrt(1 - self.alphas_cumprod[t]), x_t.shape
        )
        x_0_hat = (x_t - sqrt_1_minus_alphas_cumprod_t * epsilon) / sqrt_alphas_cumprod_t
        return x_0_hat

    def _get_epsilon_from_model(
        self, x_t: torch.Tensor, t, model_output: torch.Tensor
    ) -> torch.Tensor:
        t = t.view(-1)
        if self.prediction_type == "epsilon":
            return model_output
        elif self.prediction_type == "x0":
            sqrt_alphas_cumprod_t = self._expand(self.alphas_cumprod[t] ** 0.5, x_t.shape)
            sqrt_1_minus_alphas_cumprod_t = self._expand(
                (1 - self.alphas_cumprod[t]) ** 0.5, x_t.shape
            )
            return (x_t - sqrt_alphas_cumprod_t * model_output) / sqrt_1_minus_alphas_cumprod_t
        elif self.prediction_type == "v":
            sqrt_alphas_cumprod_t = self._expand(self.alphas_cumprod[t] ** 0.5, x_t.shape)
            sqrt_1_minus_alphas_cumprod_t = self._expand(
                (1 - self.alphas_cumprod[t]) ** 0.5, x_t.shape
            )
            return sqrt_alphas_cumprod_t * model_output + sqrt_1_minus_alphas_cumprod_t * x_t

    @torch.no_grad()
    def _sample_template(
        self,
        sample_fn: Callable,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        condition: torch.Tensor | None = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        guidance_scale: float = 1.0,
        reverse: bool = False,
        return_model_outputs: bool = False,
        batch_size: int = 128,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        _device = self.betas.device
        model.to(_device)
        x_T = x_T.to(_device)
        n_generations = 1
        n_conditions = x_T.size(0)
        spatial_dims = x_T.shape[1:]

        if condition is not None:
            condition = apply_recursively(condition, lambda x: x.to(_device))

        timesteps = self.timesteps
        if reverse:
            timesteps = timesteps.flip(0)
        n_timesteps = len(timesteps)

        if multiple_gen_per_condition:
            n_generations, n_conditions = x_T.size(0), x_T.size(1)
            spatial_dims = x_T.shape[2:]
            x_T = x_T.transpose(0, 1).flatten(0, 1)
            if condition is not None:
                condition = apply_recursively(
                    condition, lambda x: x.repeat_interleave(repeats=n_generations, dim=0)
                )

        x_T_iter = batching(x_T, batch_size, dim=0)
        if condition is not None:
            condition_iter = batching(condition, batch_size, dim=0)
        else:
            condition_iter = itertools.repeat(None)

        out_shape = (
            (n_timesteps, x_T.size(0), *spatial_dims)
            if keep_record
            else (x_T.size(0), *spatial_dims)
        )
        xs = torch.empty(out_shape, device=_device, dtype=x_T.dtype)
        vs = (
            torch.empty(out_shape, device=_device, dtype=x_T.dtype)
            if return_model_outputs
            else None
        )

        current_idx = 0
        for x_T_batch, cond_batch in zip(x_T_iter, condition_iter):
            batch_n = x_T_batch.size(0)
            x = x_T_batch

            for j, t in enumerate(timesteps):
                _t = torch.full((batch_n, 1), t, device=_device, dtype=torch.int64)
                if guidance_scale != 1.0 and cond_batch is not None:
                    out_cond = model(x, self.scale_timesteps(_t), cond_batch)
                    out_uncond = model(x, self.scale_timesteps(_t), None)
                    out = out_uncond + guidance_scale * (out_cond - out_uncond)
                else:
                    out = model(x, self.scale_timesteps(_t), cond_batch)

                step_idx = (n_timesteps - 1 - j) if reverse else j
                x = sample_fn(x, _t, out, step_idx=step_idx, generator=generator)
                if keep_record:
                    xs[j, current_idx : current_idx + batch_n] = x
                    if return_model_outputs:
                        vs[j, current_idx : current_idx + batch_n] = self._get_epsilon_from_model(
                            x, _t, out
                        )

            if not keep_record:
                xs[current_idx : current_idx + batch_n] = x
                if return_model_outputs:
                    vs[current_idx : current_idx + batch_n] = self._get_epsilon_from_model(
                        x, _t, out
                    )

            current_idx += batch_n

        def _revert_shape(tensor_to_reshape: torch.Tensor):
            return (
                tensor_to_reshape.view(
                    n_timesteps if keep_record else 1,
                    n_conditions,
                    n_generations if multiple_gen_per_condition else 1,
                    *spatial_dims,
                )
                .movedim((0, 1, 2), (1, 2, 0))
                .squeeze(1)
            )

        xs = _revert_shape(xs)
        if return_model_outputs:
            vs = _revert_shape(vs)

        if not multiple_gen_per_condition:
            xs = xs.squeeze(0)
            if return_model_outputs:
                vs = vs.squeeze(0)

        return (xs, vs) if return_model_outputs else xs

    def _threshold_sample(self, x_0_hat: torch.Tensor) -> torch.Tensor:
        dtype = x_0_hat.dtype
        batch_size, *remaining = x_0_hat.shape
        # flatten to (batch, rest) for quantile
        x = x_0_hat.float().reshape(batch_size, -1)
        s = torch.quantile(x.abs(), self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1.0, max=self.sample_max_value).unsqueeze(1)
        x = torch.clamp(x, -s, s) / s
        return x.reshape(batch_size, *remaining).to(dtype)

    def _set_timesteps(self, num_inference_steps=None, timestep_spacing=None):
        spacing = timestep_spacing or self.timestep_spacing
        n = num_inference_steps or self.num_timesteps

        if spacing == "linspace":
            timesteps = torch.linspace(0, self.num_timesteps - 1, n).round().long()
        elif spacing == "leading":
            step_ratio = self.num_timesteps // n
            timesteps = (torch.arange(0, n) * step_ratio).round().long()
        elif spacing == "trailing":
            step_ratio = self.num_timesteps / n
            timesteps = torch.arange(self.num_timesteps, 0, -step_ratio).round().long() - 1
        else:
            raise ValueError(f"Unknown timestep_spacing: {spacing}")

        timesteps = timesteps.unique(sorted=True).flip(0)  # descending: [T-1, ..., 0]

        # for reverse denoising: prev[j] = timesteps[j+1], -1 at end
        prev_timesteps = torch.cat([timesteps[1:], torch.tensor([-1])])

        # for forward inversion: next[j] = timesteps[j-1], -1 at start
        next_timesteps = torch.cat([torch.tensor([-1]), timesteps[:-1]])

        alphas_cumprod_inference_prev = torch.where(
            prev_timesteps >= 0,
            self.alphas_cumprod[prev_timesteps.clamp(min=0)],
            torch.ones(len(prev_timesteps)),
        )

        alphas_cumprod_inference_next = torch.where(
            next_timesteps >= 0,
            self.alphas_cumprod[next_timesteps.clamp(min=0)],
            torch.zeros(len(next_timesteps)),  # alphabar at "t+1 past T" = 0
        )

        device = self.alphas_cumprod.device  # match existing buffer device
        self.register_buffer("timesteps", timesteps.to(device))
        self.register_buffer("prev_timesteps", prev_timesteps.to(device))
        self.register_buffer("next_timesteps", next_timesteps.to(device))
        self.register_buffer(
            "alphas_cumprod_inference_prev", alphas_cumprod_inference_prev.float().to(device)
        )
        self.register_buffer(
            "alphas_cumprod_inference_next", alphas_cumprod_inference_next.float().to(device)
        )
