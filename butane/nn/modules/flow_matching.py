from typing import Optional, Callable, Union, Tuple
import torch


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
        x_t = (1 - (1 - self._sigma) * t) * x0 + (t * x1)
        u_t = x1 - x0 * (1 - self._sigma)
        return x_t, u_t

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timesteps: int,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        model.eval()

        t_span = torch.linspace(0, 1, timesteps).to(self._dummy_param.device)
        if condition is not None:
            condition = condition.to(self._dummy_param.device)
        x = x.to(self._dummy_param.device)
        dt = t_span[1:] - t_span[:-1]
        t = t_span[0]

        for step in range(0, len(t_span)-1):
            v_t = model(x, t.repeat(x.size(0)).reshape(-1,1), condition)
            x = x + v_t * dt[step]
            t += dt[step]
        model.train()
        return x
