from typing import List
import torch


# https://arxiv.org/pdf/2001.06782
def pcgrad(losses: List[torch.Tensor], optimizer: torch.optim.Optimizer, model: torch.nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    num_tasks = len(losses)

    # Compute per-task gradients
    task_grads = []
    for loss in losses:
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        task_grads.append(grads)

    # Project conflicting gradients
    projected_grads = []
    for i in range(num_tasks):
        grad_i = [g.clone() for g in task_grads[i]]
        for j in range(num_tasks):
            if i == j:
                continue
            grad_j = task_grads[j]

            # Flatten once for efficient dot products
            dot_product = sum(torch.dot(g_i.flatten(), g_j.flatten()) for g_i, g_j in zip(grad_i, grad_j))
            if dot_product < 0:
                norm_sq = sum(torch.sum(g_j ** 2) for g_j in grad_j)
                if norm_sq > 1e-8:
                    factor = dot_product / norm_sq
                    for k in range(len(grad_i)):
                        grad_i[k] -= factor * grad_j[k]
        projected_grads.append(grad_i)

    # Average the projected gradients and assign to params
    optimizer.zero_grad()
    for p_i, p in enumerate(params):
        p.grad = sum(g_i[p_i] for g_i in projected_grads) / num_tasks

    return
