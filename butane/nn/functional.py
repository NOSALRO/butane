from typing import Optional, List
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

def fgsm(x: torch.Tensor, epsilon: float, clip_range: Optional[List[int]] = None) -> torch.Tensor:
    # Fast Gradient Sign Method
    grad = x.grad
    petrurbated_x = (x - epsilon * grad.sign()).detach()
    return petrurbated_x.clip(clip_range[0], clip_range[1]) if clip_range is not None else petrurbated_x

# def pgm(x: torch.Tensor, epsilon: float, clip_range: Optional[List[int]] = None) -> torch.Tensor:
#     # Fast Gradient Sign Method
#     grad = x.grad
#     petrurbated_x = (x - epsilon * grad.sign()).detach()
#     return petrurbated_x.clip(clip_range[0], clip_range[1]) if clip_range is not None else petrurbated_x
