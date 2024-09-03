import math
import torch
import butane

def drop(dataset: butane.data.Dataset, prec: float) -> None:
    prec = 1.0 - prec

    if prec == 0:
        dataset.data = torch.tensor([])
        if dataset.targets.numel() != 0:
            dataset.targets = torch.tensor([])
        return

    numel_to_keep = int(math.floor(len(dataset) * prec))
    idx_to_keep = torch.randperm(len(dataset))[:numel_to_keep]

    dataset.data = dataset.data[idx_to_keep]
    if dataset.targets.numel() != 0:
        dataset.targets = dataset.targets[idx_to_keep]

def drop_to_max_size(dataset: butane.data.Dataset, max_size: int):
    if max_size >= len(dataset):
        return
    idx_to_keep = torch.randperm(len(dataset))[:max_size]
    dataset.data = dataset.data[idx_to_keep]
    if dataset.targets.numel() != 0:
        dataset.targets = dataset.targets[idx_to_keep]