from typing import Optional, Callable, Union, Tuple, List
import warnings
import copy
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

def drop_to_max_size(
    X: Union[butane.data.Dataset, torch.Tensor],
    max_size: int,
    respect_targets: bool = False,
    y: Optional[torch.Tensor] = None
) -> None:

    if isinstance(X, butane.data.Dataset):
        data = X.data
        targets = X.targets if X.targets.numel() != 0 else None
    elif isinstance(X, torch.Tensor):
        data = X
        targets = y

    if max_size >= len(data):
        return X, y
    if respect_targets and targets is not None and targets.dim() > 1:
        warnings.warn("butane.data.ops.drop_to_max_size: Targets are not class labels or they were not provied, respect_targets is now False", UserWarning)
        respect_targets = False
    if not respect_targets:
        idx_to_keep = torch.randperm(len(data))[:max_size]
        data = data[idx_to_keep]
        if targets is not None:
            targets = targets[idx_to_keep]
    else:
        unique_targets = torch.unique(targets)
        instances_per_target = max_size // len(unique_targets)
        new_data, new_targets = [], []
        for _t in unique_targets:
            _selected_data = data[targets == _t]
            _selected_data = _selected_data[torch.randperm(_selected_data.size(0))[:instances_per_target]]
            new_data.append(_selected_data)
            new_targets.append(
                torch.full(
                    size=(_selected_data.size(0),),
                    fill_value=_t,
                    dtype=targets.dtype,
                    device=targets.device
                ))
        new_data = torch.cat(new_data, dim=0)
        new_targets = torch.cat(new_targets, dim=0)
        shuffle = torch.randperm(new_data.size(0))
        data = new_data[shuffle]
        targets = new_targets[shuffle]
    if isinstance(X, butane.data.Dataset):
        X.data = data
        if targets is not None:
            X.targets = targets
    return data, targets

def bagging(dataset: butane.data.Dataset, n_learners: int) -> List[butane.data.Dataset]:
    indexes = torch.randperm(dataset.data.size(0))
    chunks = torch.chunk(indexes, n_learners)
    _bagged_datasets = []
    for chunk in chunks:
        _cp_ds = copy.deepcopy(dataset)
        _cp_ds.remove(chunk)
        _bagged_datasets.append(_cp_ds)
    return _bagged_datasets

# TODO: Better implementation
def trim(dataset: butane.data.Dataset, threshold) -> butane.data.Dataset:
    data = dataset.data
    if data.dim() == 2:
        data = data[..., None]
    res_dims = torch.arange(data.dim())[2:]
    diffs = torch.abs(data[:, 1:, :] - data[:, :-1, :])

    change_measure = diffs.mean(dim=(0, *res_dims))
    active_mask = change_measure > threshold
    if not active_mask.any():
        return dataset

    nonzero_indices = torch.nonzero(active_mask).squeeze()

    first_diff_index = nonzero_indices[0].item()
    last_diff_index  = nonzero_indices[-1].item()

    start_index = first_diff_index
    end_index = last_diff_index + 1  # plus one to get the corresponding later time step
    dataset.data = data[:, start_index:end_index + 1, ...]

    return dataset
