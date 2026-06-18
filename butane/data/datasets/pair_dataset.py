import copy

import torch

from ..transforms import Transforms
from .dataset import Dataset


class PairDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor | None = None,
        data_pair: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        targets_pair: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
        on_demand_device_load: bool = False,
        return_tuple: bool = False,
        device: str = "cpu",
    ) -> None:

        super().__init__()
        self.data = torch.tensor([]) if data is None else data.detach().clone()
        self.targets = torch.tensor([]) if targets is None else targets.detach().clone()
        self.data_pair = torch.tensor([]) if data is None else data_pair.detach().clone()
        self.targets_pair = torch.tensor([]) if targets is None else targets_pair.detach().clone()
        self._device = torch.device(device)
        self._transforms = None
        self._pair_transforms = None
        self._on_demand_device_load = on_demand_device_load
        self._return_tuple = return_tuple
        self._deterministic = deterministic

        if not self._on_demand_device_load:
            self.data, self.data_pair = self.data.to(self._device), self.data_pair.to(self._device)
            self.targets, self.targets_pair = (
                self.targets.to(self._device),
                self.targets_pair.to(self._device),
            )

        self._set_up_pairing()

    def __getitem__(
        self, idx: int | list[int]
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        assert self.data is not None and self.data_pair is not None, "Data or pairing data is None."
        device = self._device if not self._on_demand_device_load else "cpu"

        _single_sample = isinstance(idx, int)
        idx_tensor = self._prepare_idx(idx).to(device)

        if self._has_targets() and self._has_targets_pair():
            row_indices = self._anchor_row_idx[idx_tensor]
            start_i = self._start_idx[row_indices]
            end_i = self._end_idx[row_indices]
            lengths = end_i - start_i

            if self._deterministic:
                hashed_offsets = (idx_tensor * 73856093) % lengths
                pair_idx = self._target_map_csr[start_i + hashed_offsets]
            else:
                lengths = end_i - start_i
                offsets = (torch.rand(lengths.shape, device=device) * lengths).long()
                offsets = torch.clamp(offsets, max=lengths - 1)
                pair_idx = self._target_map_csr[start_i + offsets]
        else:
            pair_idx = torch.randint(0, self.data_pair.size(0), size=(len(idx_tensor),), device=device)

        _data = self.data[idx_tensor]
        _data_pair = self.data_pair[pair_idx]
        if self._on_demand_device_load:
            _data = _data.to(self._device)
            _data_pair = _data_pair.to(self._device)

        if self._transforms is not None:
            _data = self._transforms(_data)
        if self._pair_transforms is not None:
            _data_pair = self._pair_transforms(_data_pair)

        if _single_sample:
            _data = _data.squeeze(0)
            _data_pair = _data_pair.squeeze(0)
        return self._convert_to_tuple(dict(data=_data, targets=_data_pair))

    def split(
        self,
        percentage: float,
        generator: torch.Generator | None = None,
    ) -> object | None:
        if percentage == 0:
            return None

        num_samples = self.data.size(0)
        indices = torch.randperm(num_samples, generator=generator)
        split_idx = int(percentage * num_samples)

        split_1_idx = indices[:split_idx]
        split_2_idx = indices[split_idx:]

        split_2_data = self.data[split_2_idx]
        split_2_data_pair = (
            self.data_pair[split_2_idx] if self.data_pair.size(0) == num_samples else self.data_pair
        )

        self.data = self.data[split_1_idx]
        if self.data_pair.size(0) == num_samples:
            self.data_pair = self.data_pair[split_1_idx]

        if self._has_targets():
            split_2_targets = self.targets[split_2_idx]
            self.targets = self.targets[split_1_idx]
        else:
            split_2_targets = None

        if self._has_targets_pair() and self.targets_pair.size(0) == num_samples:
            split_2_targets_pair = self.targets_pair[split_2_idx]
            self.targets_pair = self.targets_pair[split_1_idx]
        else:
            split_2_targets_pair = self.targets_pair if self._has_targets_pair() else None

        if self._has_targets() and self._has_targets_pair():
            self._set_up_pairing()

        splitted_ds = PairDataset(
            data=split_2_data,
            data_pair=split_2_data_pair,
            targets=split_2_targets,
            targets_pair=split_2_targets_pair,
            on_demand_device_load=self._on_demand_device_load,
            device=str(self._device),
            return_tuple=self._return_tuple,
        )
        splitted_ds.set_transforms(self._transforms, self._pair_transforms)
        return splitted_ds

    def flatten(self, dim: int = 1) -> None:
        self.data = self.data.flatten(start_dim=dim)
        self.data_pair = self.data_pair.flatten(start_dim=dim)

    def to(self, device: torch.device):
        if not self._on_demand_device_load:
            self.data = self.data.to(device)
            self.data_pair = self.data_pair.to(device)
            if self._has_targets():
                self.targets = self.targets.to(device)
            if self._has_targets_pair():
                self.targets_pair = self.targets_pair.to(device)
            if self._has_targets_pair() and self._has_targets():
                self._target_map_csr = self._target_map_csr.to(device)
                self._start_idx = self._start_idx.to(device)
                self._end_idx = self._end_idx.to(device)
                self._anchor_row_idx = self._anchor_row_idx.to(device)
        self._device = device
        return self

    def set_transforms(
        self,
        transforms: Transforms | None = None,
        pair_transforms: Transforms | None = None,
    ) -> None:
        if transforms is not None:
            self._transforms = torch.vmap(transforms)
        if pair_transforms is not None:
            self._pair_transforms = torch.vmap(pair_transforms)

    def transforms(self) -> tuple[Transforms]:
        return self._transforms, self._pair_transforms

    def save(self, filepath: str) -> None:
        base_path = filepath.rsplit(".", 1)[0]
        torch.save(self.data.cpu().detach(), f"{base_path}_data.pt")
        torch.save(self.data_pair.cpu().detach(), f"{base_path}_data_pair.pt")
        if self._has_targets():
            torch.save(self.targets.cpu().detach(), f"{base_path}_targets.pt")
        if self._has_targets_pair():
            torch.save(self.targets_pair.cpu().detach(), f"{base_path}_targets_pair.pt")

    def set(
        self,
        *,
        data: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        data_pair: torch.Tensor | None = None,
        targets_pair: torch.Tensor | None = None,
    ) -> None:
        if data is not None:
            self.data = data.detach().clone()
        if data_pair is not None:
            self.data_pair = data_pair.detach().clone()
        if targets is not None:
            self.targets = targets.detach().clone()
        if targets_pair is not None:
            self.targets_pair = targets_pair.detach().clone()

        if not self._on_demand_device_load:
            self.data, self.data_pair = self.data.to(self._device), self.data_pair.to(self._device)
            self.targets, self.targets_pair = (
                self.targets.to(self._device),
                self.targets_pair.to(self._device),
            )

        self._set_up_pairing()

    # TODO: Implement append
    def append(self, data: torch.Tensor, targets: torch.Tensor | None = None) -> None: ...

    def remove(self, idxs):
        mask = torch.ones((self.data.size(0),), dtype=bool)
        mask[idxs] = False
        self.data = self.data[mask]
        if self._has_targets():
            self.targets = self.targets[mask]

    def _split(
        self,
        percentage: float,
        data: torch.Tensor,
        targets: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        split_1_targets = None
        split_2_targets = None
        indices = torch.randperm(data.size(0), generator=generator)
        num_el_after_split = int(percentage * data.size(0))

        split_2_data = data[indices[num_el_after_split:]]
        split_1_data = data[indices[:num_el_after_split]]

        if self._has_targets() and self._has_targets_pair():
            split_1_targets = targets[indices[:num_el_after_split]]
            split_2_targets = targets[indices[num_el_after_split:]]
        return (split_1_data, split_1_targets), (split_2_data, split_2_targets)

    def _set_up_pairing(self):
        # Clean samples that cannot be matched
        if not (self._has_targets() or self._has_targets_pair()):
            return
        unique_targets_pair = torch.unique(self.targets_pair)
        mask = torch.isin(self.targets.view(-1), unique_targets_pair)

        if not mask.all():
            self.data = self.data[mask]
            self.targets = self.targets[mask]

        # Make a target-agnostic indexing map
        _target_loc, self._anchor_row_idx = torch.unique(self.targets.view(-1), return_inverse=True)

        # Create a map of indices for all the possible samples per class
        num_elems = [0]
        self._target_map_csr = []
        for i in range(_target_loc.size(0)):
            pair = (self.targets_pair == _target_loc[i]).nonzero().flatten()
            self._target_map_csr.append(pair)
            num_elems.append(pair.numel())
        num_elems = torch.tensor(num_elems).cumsum(0)
        self._start_idx = num_elems[:-1].long()
        self._end_idx = num_elems[1:].long()
        self._target_map_csr = torch.cat(self._target_map_csr).long()

        if not self._on_demand_device_load:
            self._start_idx = self._start_idx.to(self._device)
            self._end_idx = self._end_idx.to(self._device)
            self._target_map_csr = self._target_map_csr.to(self._device)

    def _prepare_idx(self, idx: int | list[int] | tuple[int] | slice | torch.Tensor) -> torch.Tensor:
        if isinstance(idx, int):
            return torch.tensor([idx]).long()
        elif isinstance(idx, (list, tuple)):
            return torch.tensor(idx).long()
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.data.size(0))
            return torch.arange(start, stop, step, dtype=torch.long)
        elif isinstance(idx, torch.Tensor):
            return idx.long()

    def _has_targets(self):
        return self.targets.numel() > 0

    def _has_targets_pair(self):
        return self.targets_pair.numel() > 0

    def _convert_to_tuple(
        self,
        outputs: dict[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor] | dict[str, torch.Tensor]:
        if self._return_tuple:
            return outputs["data"], outputs["targets"]
        return outputs
