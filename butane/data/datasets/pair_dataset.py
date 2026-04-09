import copy
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..transforms import Transforms
from .dataset import Dataset


class PairDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        data_pair: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_pair: Optional[torch.Tensor] = None,
        *,
        on_demand_device_load: bool = False,
        return_tuple: bool = False,
        device: torch.device = "cpu",
    ) -> None:

        super().__init__()
        self.data = torch.tensor([]) if data is None else data.detach().clone()
        self.targets = torch.tensor([]) if targets is None else targets.detach().clone()
        self.data_pair = torch.tensor([]) if data is None else data_pair.detach().clone()
        self.targets_pair = torch.tensor([]) if targets is None else targets_pair.detach().clone()
        self._device = torch.device(device)
        self._transforms, self._vectorized_transforms = None, None
        self._pair_transforms, self._vectorized_pair_transforms = None, None
        self._on_demand_device_load = on_demand_device_load
        self._return_tuple = return_tuple
        if not self._on_demand_device_load:
            self.data, self.data_pair = self.data.to(self._device), self.data_pair.to(self._device)
            self.targets, self.targets_pair = (
                self.targets.to(self._device),
                self.targets_pair.to(self._device),
            )
        self._set_up_pairing()

    def _set_up_pairing(self):
        if self._has_targets() and self._has_targets_pair():
            unique_targets_pair = torch.unique(self.targets_pair)
            mask = torch.isin(self.targets.view(-1), unique_targets_pair)

            if not mask.all():
                self.data = self.data[mask]
                self.targets = self.targets[mask]
            self._target_loc, indices = torch.unique(self.targets, return_inverse=True)
            max_num_of_instances = torch.bincount(self.targets_pair.flatten()).max().item()

            self._target_map = torch.full(
                (self._target_loc.size(0), max_num_of_instances),
                fill_value=-1,
                dtype=torch.int64,
            )

            for i in range(self._target_loc.size(0)):
                pair = (self.targets_pair == self._target_loc[i]).nonzero().flatten()
                self._target_map[i] = torch.nn.functional.pad(
                    pair, (0, max_num_of_instances - pair.numel()), value=-1, mode="constant"
                )

            self._col_idx = (self._target_map != -1).sum(dim=1)
            self._anchor_row_idx = torch.searchsorted(unique_targets_pair, self.targets.view(-1))
            if not self._on_demand_device_load:
                self._target_loc = self._target_loc.to(self._device)
                self._target_map = self._target_map.to(self._device)
                self._col_idx = self._col_idx.to(self._device)
            # TODO: Clean data from the pair dataset side

    def _has_targets(self):
        return self.targets is not None and self.targets.numel() > 0

    def _has_targets_pair(self):
        return self.targets_pair is not None and self.targets_pair.numel() > 0

    # TODO: Optimize
    def __getitem__(self, idx: Union[int, List[int]]) -> Dict[str, torch.Tensor]:
        device = self._device if not self._on_demand_device_load else "cpu"
        use_vectorized_transforms = False
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            idx = torch.arange(start, stop, step, device=device)
            use_vectorized_transforms = True
        elif isinstance(idx, int):
            idx = torch.tensor([idx], device=device)
        else:  # List
            idx = torch.tensor(idx, device=device)
            use_vectorized_transforms = True

        if self._has_targets() and self._has_targets_pair():
            row_indices = self._anchor_row_idx[idx]
            limits = self._col_idx[row_indices]
            random_cols = (torch.rand(limits.numel(), device=device) * limits).long()
            pair_idx = self._target_map[row_indices, random_cols]
        else:
            pair_idx = torch.randint(0, self.data_pair.size(0), size=(len(idx),), device=device)

        _data = self.data[idx.item() if idx.numel() == 1 else idx.ravel()]
        _data_pair = self.data_pair[pair_idx.item() if pair_idx.numel() == 1 else pair_idx.ravel()]
        if self._on_demand_device_load:
            _data = _data.to(self._device)
            _data_pair = _data_pair.to(self._device)

        if self._transforms is not None:
            _data = (
                self._vectorized_transforms(_data)
                if use_vectorized_transforms
                else self._transforms(_data)
            )
        if self._pair_transforms is not None:
            _data_pair = (
                self._vectorized_pair_transforms(_data_pair)
                if use_vectorized_transforms
                else self._pair_transforms(_data_pair)
            )
        return self._convert_to_tuple(dict(data=_data, targets=_data_pair))

    def split(
        self,
        percentage: float,
        generator: Optional[torch.Generator] = None,
    ):
        if percentage == 0:
            return None
        _transforms = self._transforms
        _pair_transforms = self._pair_transforms
        (split_1_data, split_1_targets), (split_2_data, split_2_targets) = self._split(
            percentage, self.data, self.targets, generator=generator
        )
        (split_1_data_pair, split_1_targets_pair), (split_2_data_pair, split_2_targets_pair) = (
            self._split(percentage, self.data_pair, self.targets_pair, generator=generator)
        )

        self.data = split_1_data
        self.targets = split_1_targets
        self.data_pair = split_1_data_pair
        self.targets_pair = split_1_targets_pair
        self._set_up_pairing()

        splitted_ds = PairDataset(
            data=split_2_data,
            targets=split_2_targets,
            data_pair=split_2_data_pair,
            targets_pair=split_2_targets_pair,
            on_demand_device_load=self._on_demand_device_load,
            device=self._device,
            return_tuple=self._return_tuple,
        )
        splitted_ds.set_transforms(_transforms, _pair_transforms)
        return splitted_ds

    def flatten(self, dim: int = 1) -> None:
        self.data = self.data.flatten(start_dim=dim)
        self.paired_data = self.paired_data.flatten(start_dim=dim)

    def to(self, device: torch.device) -> None:
        if not self._on_demand_device_load:
            self.data = self.data.to(device)
            self.data_pair = self.data_pair.to(device)
            if self._has_targets():
                self.targets = self.targets.to(device)
            if self._has_targets_pair():
                self.targets_pair = self.targets_pair.to(device)
            if self._has_targets_pair() and self._has_targets():
                self._target_loc = self._target_loc.to(device)
                self._target_map = self._target_map.to(device)
                self._col_idx = self._col_idx.to(device)
                self._anchor_row_idx = self._anchor_row_idx.to(device)
        self._device = device
        return self

    def set_transforms(
        self,
        transforms: Optional[Transforms] = None,
        pair_transforms: Optional[Transforms] = None,
    ):
        if transforms is not None:
            self._transforms = transforms
            self._vectorized_transforms = torch.vmap(transforms)
        if pair_transforms is not None:
            self._pair_transforms = pair_transforms
            self._vectorized_pair_transforms = torch.vmap(pair_transforms)

    def transforms(self) -> Tuple[Transforms]:
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
        data: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        paired_data: Optional[torch.Tensor] = None,
        paired_targets: Optional[torch.Tensor] = None,
    ) -> None:
        self.data = data.detach().clone()
        self.paired_data = paired_data.detach().clone()
        if targets is not None:
            self.targets = targets.detach().clone()
        if paired_targets is not None:
            self.paired_targets = paired_targets.detach().clone()

    # TODO: Implement append
    def append(self, data: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None: ...

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
        targets: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
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

    def _convert_to_tuple(
        self,
        returns: dict,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        if self._return_tuple:
            data = returns["data"]
            if self._has_targets():
                target = returns["targets"]
                return data, target
            return data
        else:
            return returns
