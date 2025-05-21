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
