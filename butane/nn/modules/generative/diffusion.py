import warnings
from typing import Optional, Callable, Union
import functools
import math
import torch
from ...functional import *
from ....math.ops import *

class Diffusion(torch.nn.Module):

    def __init__(
            self,
            num_timesteps: int,
            scheduler: str = 'linear',
            *,
            scale_timesteps: bool = False,
            model_predicts_noise: bool = True,
            variance_type: str = 'fixed_large',
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.model_predicts_noise = model_predicts_noise

        self.scale_timesteps = (lambda x: x.float()) if not scale_timesteps else self._scale_timesteps_fn
        # self.unscale_timesteps = (lambda x: x.int()) if not scale_timesteps else self._scale_timesteps_fn

        _variance_options = ['fixed_large', 'fixed_small', 'learned', 'learned_range']
        assert variance_type in _variance_options, f"Options for variance are: {_variance_options}"
        self.variance_type = variance_type

        if scheduler == 'linear':
            beta = self._linear_beta_scheduler(self.num_timesteps)
        elif scheduler == 'cosine':
            beta = self._cosine_beta_scheduler(self.num_timesteps)

        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), alpha_cumprod[:-1]])
        alpha_cumprod_next = torch.cat([alpha_cumprod[1:], torch.tensor([0.0])])

        beta_tilde = ((1. - alpha_cumprod_prev) / (1 - alpha_cumprod)) * beta
        log_beta_tilde = torch.log(torch.hstack([beta_tilde[1], beta_tilde[1:]]))

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("alpha_cumprod_next", alpha_cumprod_next)
        self.register_buffer("beta_tilde", beta_tilde)
        self.register_buffer("log_beta_tilde", log_beta_tilde)

    def q_params(
        self,
        x_0: torch.Tensor,
        t: Union[torch.Tensor, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.__expand(torch.sqrt(self.alpha_cumprod[t]), x_0.shape) * x_0
        var = self.__expand(1.0 - self.alpha_cumprod[t], x_0.shape)
        return mu, var

    def q_posterior_params(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_coeff_1 = (torch.sqrt(self.alpha_cumprod_prev[t]) * self.beta[t]) / (1 - self.alpha_cumprod[t])
        mu_coeff_2 = torch.sqrt(self.alpha[t])  * (1. - self.alpha_cumprod_prev[t]) / (1 - self.alpha_cumprod[t])
        mu = self.__expand(mu_coeff_1, x_0.shape) * x_0 + self.__expand(mu_coeff_2, x_t.shape) * x_t
        logvar = self.__expand(self.log_beta_tilde[t], x_t.shape)
        var = self.__expand(self.beta_tilde[t], x_t.shape)
        return mu, var, logvar

    def p_params(
        self,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int],
        model_output:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.model_predicts_noise:
            if self.variance_type not in ['learned', 'learned_range']:
                epsilon_hat = model_output
                if self.variance_type == 'fixed_small':
                    var = self.__expand(self.beta_tilde[t], x_t.shape)
                    logvar = self.__expand(self.log_beta_tilde[t], x_t.shape)
                elif self.variance_type == 'fixed_large':
                    _tmp_beta = torch.hstack([self.beta_tilde[0], self.beta[1:]])
                    var = self.__expand(_tmp_beta[t], x_t.shape)
                    logvar = torch.log(self.__expand(_tmp_beta[t], x_t.shape))
            else:
                epsilon_hat, logvar_hat = torch.chunk(model_output, chunks=2, dim=1)
                if self.variance_type == 'learned':
                    logvar = logvar_hat
                    var = torch.exp(logvar)
                elif self.variance_type == 'learned_range':
                    log_beta_t = self.__expand(torch.log(self.beta[t]), logvar_hat.shape)
                    log_beta_tilde_t = self.__expand(self.log_beta_tilde[t], logvar_hat.shape)
                    logvar_hat_normalized = (logvar_hat + 1) / 2
                    logvar = (logvar_hat_normalized * log_beta_t) + ((1 - logvar_hat_normalized) * log_beta_tilde_t)
                    var = torch.exp(logvar)

            # This is very unstable. Generated data can drift away from support.
            # beta_t = self.__expand(self.beta[t], x_t.shape)
            # sqrt_one_minus_alpha_cumprod_t = self.__expand(torch.sqrt(1.0 - self.alpha_cumprod[t]), x_t.shape)
            # sqrt_alpha_t = self.__expand(torch.sqrt(self.alpha[t]), x_t.shape)
            # mu = (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * epsilon_hat) * (1 / sqrt_alpha_t)
            x_0_hat = self._derive_x_0_from_epsilon(x_t, t, epsilon_hat, clip=True)
            mu, _, _ = self.q_posterior_params(x_0_hat, x_t, t)
            return mu, var, logvar, x_0_hat

    def forward(
        self,
        x_0: torch.Tensor,
        t: Union[torch.Tensor, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, var = self.q_params(x_0, t)
        epsilon = torch.randn_like(x_0)
        return mu + torch.sqrt(var) * epsilon, epsilon

    def reverse(
        self,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int],
        model_output:  torch.Tensor,
    ) -> torch.Tensor:
        mu, _, logvar, _ = self.p_params(x_t, t, model_output)
        z = torch.randn_like(x_t)
        z_mask = torch.ones_like(z)
        z_mask[t.flatten() == 0] *= 0.
        x_t_minus_1 = mu + torch.exp(0.5 * logvar) * z * z_mask
        return x_t_minus_1

    def sample_timesteps(self, n: int) -> torch.Tensor:
        t = torch.randint(0, self.num_timesteps, size=(n, 1), device=self.beta.device, dtype=torch.int64)
        return t

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        return_model_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._sample_template(
            sample_fn=self.reverse,
            model=model,
            x_T=x_T,
            condition=condition,
            keep_record=keep_record,
            multiple_gen_per_condition=multiple_gen_per_condition,
            timestep_stride=1,
            reverse=False,
            return_model_outputs=return_model_outputs,
        )

    @torch.no_grad()
    def sample_ddim(
        self,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        timestep_stride: int = 1,
        reverse: bool = False,
        return_model_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sample_fn = self._ddim if not reverse else functools.partial(self._ddim, reverse=True)
        return self._sample_template(
            sample_fn=sample_fn,
            model=model,
            x_T=x_T,
            condition=condition,
            keep_record=keep_record,
            multiple_gen_per_condition=multiple_gen_per_condition,
            timestep_stride=timestep_stride,
            reverse=reverse,
            return_model_outputs=return_model_outputs,
        )

    def simple_loss(self, x_true, x_hat) -> torch.Tensor:
        return (x_hat - x_true).pow(2).mean()

    def vb_loss(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int],
        model_output:  torch.Tensor,
    ) -> torch.Tensor:
        q_mu, _, q_logvar = self.q_posterior_params(x_0, x_t, t)
        p_mu, _, p_logvar, _ = self.p_params(x_t, t, model_output)
        p_mu = p_mu.detach()
        kldiv = kl_div_gaussians(mu1=q_mu, mu2=p_mu, logvar1=q_logvar, logvar2=p_logvar)
        kldiv = apply_around_dim(torch.mean, kldiv, dims=0)

        discrete_nll = -self.__discretized_gaussian_nll(x_0, mu=p_mu, logvar=p_logvar)
        discrete_nll = apply_around_dim(torch.mean, discrete_nll, dims=0)

        vb_loss = torch.where((t.flatten() == 0), discrete_nll, kldiv)
        return vb_loss.mean()

    def loss(
        self,
        x_true: torch.Tensor,
        model_output:  torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        if self.variance_type in ['learned', 'learned_range']:
            assert model_output.size(1) == (2 * x_true.size(1)), "Variance is learnt, double the model output at dim=1"
            x_hat, _ = torch.chunk(model_output, chunks=2, dim=1)
            L_simple = self.simple_loss(x_true, x_hat)
            L_vb = self.vb_loss(x_0, x_t, t, model_output)
            return L_simple + 1e-03 * L_vb
        else:
            return self.simple_loss(x_true, model_output)

    @torch.no_grad()
    def _ddim(
        self,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int],
        model_output: torch.Tensor,
        eta: float = 0.0,
        reverse: bool = False
    ) -> torch.Tensor:

        epsilon_hat = self.__get_epsilon_from_model(model_output)
        _, _, _, x_0_hat = self.p_params(x_t, t, model_output)

        if not reverse:
            alpha_cumprod_t = self.__expand(self.alpha_cumprod[t], x_t.shape)
            alpha_cumprod_prev_t = self.__expand(self.alpha_cumprod_prev[t], x_t.shape)
            sigma = eta * (torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev_t))
            mu = (x_0_hat * torch.sqrt(alpha_cumprod_prev_t)) + (torch.sqrt(1 - alpha_cumprod_prev_t - sigma ** 2) * epsilon_hat)
            noise = torch.randn_like(x_t)
            noise_mask = torch.ones_like(noise)
            noise_mask[t.flatten() == 0] *= 0.
            x_t_minus_1 = mu + sigma * noise * noise_mask
            return x_t_minus_1
        else:
            alpha_cumprod_next_t = self.__expand(self.alpha_cumprod_next[t], x_t.shape)
            x_t_plus_1 = torch.sqrt(alpha_cumprod_next_t)  * x_0_hat + torch.sqrt(1 - alpha_cumprod_next_t) * epsilon_hat
            return x_t_plus_1

    def _linear_beta_scheduler(self, num_timesteps: int) -> torch.Tensor:
        scale = 1000.0 / num_timesteps
        beta_start = scale * 1e-04
        beta_end = scale * 0.02
        return torch.linspace(
            start=beta_start,
            end=beta_end,
            steps=num_timesteps,
        )

    def _cosine_beta_scheduler(self, num_timesteps: int, s: float = 0.008):
        beta = []
        f_t = lambda t: (
            math.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
        )
        for t in range(num_timesteps):
            t1 = t / num_timesteps
            t2 = (t + 1) / num_timesteps
            beta.append(min(1 - (f_t(t2) / f_t(t1)), 0.999))
        return torch.tensor(beta)

    def _scale_timesteps_fn(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() * (1000.0 / self.num_timesteps)

    def _unscale_timesteps_fn(self, t: torch.Tensor) -> torch.Tensor:
        return (t * (self.num_timesteps / 1000.0)).int()

    @staticmethod
    def __expand(x, shape: tuple):
        while len(x.size()) != len(shape):
            x = x[..., None]
        return x

    def _derive_x_0_from_epsilon(
        self,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int],
        epsilon: torch.Tensor,
        clip: bool = False,
    ) -> torch.Tensor:
        sqrt_alpha_cumprod_t = self.__expand(torch.sqrt(self.alpha_cumprod[t]), x_t.shape)
        sqrt_1_minus_alpha_cumprod_t = self.__expand(torch.sqrt(1 - self.alpha_cumprod[t]), x_t.shape)
        x_0_hat = (x_t - sqrt_1_minus_alpha_cumprod_t * epsilon)  / sqrt_alpha_cumprod_t
        if clip:
            return x_0_hat.clip(-1,1)
        return x_0_hat

    @torch.no_grad()
    def _sample_template(
        self,
        sample_fn: Callable,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        timestep_stride: int = 1,
        reverse: bool = False,
        return_model_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        model.eval()
        if condition is not None:
            condition = condition.to(self.beta.device)

        if not multiple_gen_per_condition:
            x_T = x_T.unsqueeze(0)

        if keep_record:
            generated_samples = torch.empty(x_T.size(0), self.num_timesteps // timestep_stride, *x_T.size()[1:])
            model_outputs = torch.empty(x_T.size(0), self.num_timesteps // timestep_stride, *x_T.size()[1:])
        else:
            generated_samples = torch.empty_like(x_T)
            model_outputs = torch.empty_like(x_T)

        x_T = x_T.to(self.beta.device)

        timesteps = range(0, self.num_timesteps, timestep_stride)
        if not reverse:
            timesteps = reversed(timesteps)

        for i, x in enumerate(x_T):
            for j, t in enumerate(timesteps):
                _t = torch.full((x.size(0), 1), t, device=self.beta.device, dtype=torch.int64)
                out = model(x, self.scale_timesteps(_t), condition)
                x = sample_fn(x, _t, out)
                if keep_record:
                    generated_samples[i, j] = x
                    if return_model_outputs:
                        model_outputs[i, j] = self.__get_epsilon_from_model(out)
            if not keep_record:
                generated_samples[i] = x
                if return_model_outputs:
                    model_outputs[i] = self.__get_epsilon_from_model(out)
        model.train()
        if return_model_outputs:
            return generated_samples.squeeze(0), model_outputs.squeeze(0)
        else:
            return generated_samples.squeeze(0)

    def __get_epsilon_from_model(self, model_output: torch.Tensor) -> torch.Tensor:
        if self.model_predicts_noise:
            return model_output
        else:
            epsilon_hat, _ = torch.chunk(model_output, chunks=2, dim=1)
            return epsilon_hat


    def __get_epsilon_from_model(self, model_output: torch.Tensor) -> torch.Tensor:
        if self.model_predicts_noise:
            return model_output
        else:
            epsilon_hat, _ = torch.chunk(model_output, chunks=2, dim=1)
            return epsilon_hat

    @staticmethod
    def __discretized_gaussian_nll(x: torch.Tensor, *, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        assert x.shape == mu.shape == logvar.shape
        centered_x = x - mu
        inv_stdv = torch.exp(-logvar * 0.5)
        bin = (1 / 255.0)
        plus_in = inv_stdv * (centered_x + bin)
        cdf_plus = approx_cumulative_normal_function(plus_in)
        min_in = inv_stdv * (centered_x - bin)
        cdf_min = approx_cumulative_normal_function(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs

    @torch.no_grad()
    def inpainting(
        self,
        model: torch.nn.Module,
        x_T: torch.Tensor,
        x_original: torch.Tensor,
        mask: torch.Tensor,
        multiple_gen_per_mask: bool = False,
        keep_record: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        model.eval()
        if not multiple_gen_per_mask:
            x_T = x_T.unsqueeze(0)

        if keep_record:
            generated_samples = torch.empty(x_T.size(0), self.num_timesteps, *x_T[1:].size())
        else:
            generated_samples = torch.empty_like(x_T)

        x_T = x_T.to(self.beta.device)
        x_original = x_original.to(self.beta.device)
        mask = mask.to(self.beta.device)

        timesteps = reversed(range(0, self.num_timesteps, 1))
        for i, x in enumerate(x_T):
            for j, t in enumerate(timesteps):
                _t = torch.full((x.size(0), 1), t, device=self.beta.device, dtype=torch.int64)
                out = model(x, self.scale_timesteps(_t))
                x_original_diffused, _ = self.forward(x_original, t)
                x = self.reverse(x, _t, out)
                x = x_original_diffused * mask + (1 - mask) * x
                if keep_record:
                    generated_samples[i, j] = x
            if not keep_record:
                generated_samples[i] = x
        model.train()
        return generated_samples
