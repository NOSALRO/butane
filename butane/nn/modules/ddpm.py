from typing import Optional, Callable, Union, Tuple
from typing_extensions import Self
import math
import torch


class DDPM(torch.nn.Module):

    def __init__(
        self,
        timesteps: int,
        beta_limits: list[float],
        scheduler: Optional[str] = 'linear'
    ) -> None:
        super().__init__()
        self._timesteps = torch.arange(timesteps).type(torch.float32)
        self._beta_limits = beta_limits
        # trick to get module's device
        self._dummy_param = torch.nn.Parameter(torch.empty(0))
        if scheduler == 'linear':
            self.beta_linear_scheduler()

    def sample_timesteps(self, n: int) -> None:
        return torch.randint(0, self._timesteps.size(0), size=(n, 1), device=self._dummy_param.device)

    def beta_cosine_scheduler(self, s: Optional[float] = 0.008) -> None:
        timesteps = torch.arange(len(self._timesteps) + 1).float()
        alpha_hat = torch.cos(((timesteps / timesteps.size(0)) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_hat = alpha_hat / alpha_hat[0]
        self._beta = 1 - (alpha_hat[1:] / alpha_hat[:-1])
        self._beta = torch.clip(self._beta, 0.0001, 0.9999)
        self._beta = self._beta.to(self._dummy_param.device)
        self.__prepare_coefs()

    def beta_linear_scheduler(self) -> None:
        self._beta = torch.linspace(self._beta_limits[0], self._beta_limits[1], len(self._timesteps))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(x)
        sqrt_alpha_hat_t = self._sqrt_alpha_hat[t]
        sqrt_one_minus_alpha_hat_t = self._sqrt_one_minus_alpha_hat[t]

        while len(x.size()) != len(sqrt_alpha_hat_t.size()):
            sqrt_alpha_hat_t = sqrt_alpha_hat_t[...,None]
            sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat_t[...,None]

        x_t = (sqrt_alpha_hat_t * x) + (sqrt_one_minus_alpha_hat_t * eps)
        return x_t, eps

    def reverse(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha_t = self._sqrt_alpha[t]
        sqrt_one_minus_alpha_hat_t = self._sqrt_one_minus_alpha_hat[t]
        beta_t = self._beta[t]
        sqrt_beta_t = self._sqrt_beta[t]

        while len(x.size()) != len(sqrt_alpha_t.size()):
            sqrt_alpha_t = sqrt_alpha_t[...,None]
            sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat_t[...,None]
            beta_t = beta_t[..., None]
            sqrt_beta_t = sqrt_beta_t[..., None]

        if t.flatten(0)[0].item() > 1:
            return 1./sqrt_alpha_t * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * eps)) + (sqrt_beta_t * torch.randn_like(x))
        else:
            return (1./sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * eps))

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        *,
        keep_record: Optional[bool] = False
    ) -> torch.Tensor:

        model.eval()
        x = x.to(self._dummy_param.device)
        if condition is not None:
            condition = condition.to(self._dummy_param.device)
        t_template = torch.ones((x.size(0), 1)).to(self._dummy_param.device)

        if keep_record:
            _record = [x.unsqueeze(0)]

        for i in reversed(self._timesteps):
            t = t_template * i
            predicted_noise = model(x, t, condition)
            x = self.reverse(x, t.long(), predicted_noise)
            if keep_record:
                _record.append(x.unsqueeze(0))
        model.train()
        return x if not keep_record else torch.vstack(_record)

    def to(self, *args, **kwargs) -> Self:
        super().to(*args, **kwargs)
        self._beta = self._beta.to(self._dummy_param.device)
        self.__prepare_coefs()
        return self

    def __prepare_coefs(self):
        self._alpha = 1 - self._beta
        self._alpha_hat = torch.cumprod(self._alpha, dim=0)
        self._sqrt_alpha = torch.sqrt(self._alpha)
        self._sqrt_alpha_hat = torch.sqrt(self._alpha_hat)
        self._sqrt_one_minus_alpha_hat = torch.sqrt(1. - self._alpha_hat)
        self._sqrt_beta = torch.sqrt(self._beta)

