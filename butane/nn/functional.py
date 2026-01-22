from typing import Optional, List, Callable
import torch


def gaussian(x: torch.Tensor) -> torch.Tensor:
    return (-x.square()).exp()


def squashing(x: torch.Tensor) -> torch.Tensor:
    return (9 / 8.0 * torch.sin(x)) + (1 / 8.0 * torch.sin(3.0 * x))


def scaled_tanh(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * torch.tanh(x)


# Losses
def pinball_loss(input, target, alpha, reduction="mean") -> torch.Tensor:
    assert reduction in ["mean", "sum", "none"], "Reduction can be mean, sum or none"

    loss = torch.max(alpha * (target - input), (1 - alpha) * (input - target))

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

def kl_div_gaussians(mu1, logvar1, mu2, logvar2) -> torch.Tensor:
    k = -1.0
    log_det_ratio = logvar2 - logvar1
    trace_term = (logvar1 - logvar2).exp()
    mahalanobis_distance = (mu1 - mu2).pow(2) * torch.exp(-logvar2)
    kl_div = 0.5 * (log_det_ratio + k + mahalanobis_distance + trace_term)
    return kl_div

def fgm(
    x: torch.Tensor,
    J: Callable,
    epsilon: float,
    clip_range: Optional[List[float]] = None,
    use_sign: bool = False
) -> torch.Tensor:

    x_prime = x.detach().clone().requires_grad_(True)
    loss = J(x_prime)
    grad = torch.autograd.grad(loss, x_prime, retain_graph=False, create_graph=False)[0]
    with torch.no_grad():
        if use_sign:
            perturbed_x = x_prime + epsilon * grad.sign()
        else:
            grad_norm = grad.flatten(1).norm(dim=-1)
            while grad_norm.dim() != grad.dim():
                grad_norm = grad_norm[..., None]
            perturbed_x = x_prime + epsilon * grad / (grad_norm + 1e-12)
        if clip_range is not None:
            perturbed_x = torch.clamp(perturbed_x, min=clip_range[0], max=clip_range[1])
    return perturbed_x.detach()

def pgm(
    x: torch.Tensor,
    J: Callable,
    epsilon: float,
    steps: int = 10,
    clip_range: Optional[List[float]] = None,
    use_sign: bool = False,
    random_start: bool = True,
) -> torch.Tensor:

    alpha = epsilon / steps
    x_init = x.detach().clone()

    if random_start:
        noise = torch.empty_like(perturbed_x).uniform_(-epsilon, epsilon)
        perturbed_x = x_init + noise
        if clip_range is not None:
            perturbed_x = torch.clamp(perturbed_x, min=clip_range[0], max=clip_range[1])

    x_prime = x_init.clone()
    perturbed_x = x_init.clone()
    for _ in range(steps):
        x_prime = fgm(
            perturbed_x,
            J=J,
            epsilon=alpha,
            clip_range=None,
            use_sign=use_sign,
        )
        delta = (x_prime - x_init).clamp(-epsilon, epsilon)
        perturbed_x = x_init +  delta
        if clip_range is not None:
            perturbed_x = torch.clamp(perturbed_x, min=clip_range[0], max=clip_range[1])
    return perturbed_x.detach()
