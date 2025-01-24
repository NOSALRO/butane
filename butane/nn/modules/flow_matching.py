from typing import Optional, Callable, Union, Tuple
import torch

from ...math.ode import *


class ConditionalFlowMatching(torch.nn.Module):

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        x_t = mu_t + self._sigma * epsilon
        u_t = x1 - x0
        return x_t, u_t

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        dims: Union[list[int,...], tuple[int], torch.Tensor],
        sample_fn: Callable,
        timesteps: int,
        condition: Optional[torch.Tensor] = None,
        keep_record: Optional[bool] = False,
        n_samples_per_condition: int = 1,
        solver: Optional[str] = 'rk4'
    ) -> torch.Tensor:

        model.eval()
        _solver = rk4
        if solver == 'euler':
            _solver = euler_explicit

        if condition is not None:
            condition = condition.to(self._dummy_param.device)

        generated = []
        for _ in range(n_samples_per_condition if condition is not None else 1):
            _record = []
            x = sample_fn(*dims)
            t_span = torch.linspace(0, 1, timesteps).to(self._dummy_param.device)
            x = x.to(self._dummy_param.device)
            if keep_record:
                _record.append(x.unsqueeze(0))
            dt = t_span[1:] - t_span[:-1]
            t = t_span[0]

            for step in range(0, len(t_span)-1):
                x = x + _solver(model, dt[step], x, t.repeat(x.size(0)).reshape(-1,1), condition)
                if keep_record:
                    _record.append(x.unsqueeze(0))
                t += dt[step]
            generated_i = torch.vstack(_record) if keep_record else x
            generated.append(generated_i.unsqueeze(0))
        model.train()
        return torch.vstack(generated).squeeze(0)
