from typing import List
import torch


# https://arxiv.org/pdf/2001.06782
def pcgrad(losses: List[torch.Tensor], optimizer: torch.optim.Optimizer, model: torch.nn.Module):
    num_tasks = len(losses)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad()

    # Step 1: Compute and store task-specific gradients
    task_grads = []
    for loss in losses:
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grad = [p.grad.detach().clone() if p.grad is not None else None for p in params]
        task_grads.append(grad)

    # Step 2: Project conflicting gradients
    projected_grads = []
    for i in range(num_tasks):
        grad_i = task_grads[i].copy()

        for j in range(num_tasks):
            if i == j:
                continue

            grad_j = task_grads[j]
            dot_product = sum(
                torch.dot(g_i.flatten(), g_j.flatten())
                for g_i, g_j in zip(grad_i, grad_j)
                if g_i is not None and g_j is not None
            )

            if dot_product < 0:
                norm_sq = sum(torch.sum(g_j**2) for g_j in grad_j if g_j is not None)
                if norm_sq > 1e-8:
                    factor = dot_product / norm_sq
                    grad_i = [
                        g_i - factor * g_j if g_i is not None and g_j is not None else g_i
                        for g_i, g_j in zip(grad_i, grad_j)
                    ]

        projected_grads.append(grad_i)

    # Step 3: Average projected gradients and set them to model
    final_grads = [torch.zeros_like(p) for p in params]
    for grads in projected_grads:
        for i, g in enumerate(grads):
            if g is not None:
                final_grads[i] += g

    for i, p in enumerate(params):
        p.grad = final_grads[i] #/ num_tasks
