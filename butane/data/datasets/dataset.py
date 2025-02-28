from typing import Optional, List, Tuple, Dict, Self, Union
import torch

from ..transforms import Transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.data = torch.tensor([]) if data is None else data.detach().clone().float()
        self.targets = torch.tensor([]) if targets is None else targets.detach().clone()
        self._has_targets = targets is not None
        self._device = torch.device('cpu')
        self._transforms, self._vectorized_transforms = None, None

    def __getitem__(self, idx: Union[int, List[int]]) -> Dict[str, torch.Tensor]:
        if isinstance(idx, int):
            if self._has_targets:
                return {
                    "data": self.data[idx] if self._transforms is None else self._transforms(self.data[idx]),
                    "targets": self.targets[idx]
                }
            else:
                return { "data": self.data[idx] if self._transforms is None else self._transforms(self.data[idx]) }
        else:
            if self._has_targets:
                return {
                    "data": self.data[idx] if self._transforms is None else self._vectorized_transforms(self.data[idx]),
                    "targets": self.targets[idx]
                }
            else:
                return { "data": self.data[idx] if self._transforms is None else self._vectorized_transforms(self.data[idx]) }

    def split(self, percentage: float):
        (split_1_data, split_1_targets), (split_2_data, split_2_targets) = self._split(percentage)
        self.__init__(split_1_data, split_1_targets)
        return Dataset(split_2_data, split_2_targets)

    def flatten(self, dim: int = 1) -> None:
        self.data = self.data.flatten(start_dim=dim)

    def to(self, device: torch.device) -> None:
        self.data = self.data.to(device)
        if self._has_targets:
            self.targets = self.targets.to(device)
        self._device = device

    def save(self, filepath: str) -> None:
        base_path = filepath.rsplit('.', 1)[0]
        torch.save(self.data.cpu().detach(), f"{base_path}_data.pt")
        if self._has_targets:
            torch.save(self.targets.cpu().detach(), f"{base_path}_targets.pt")

    def set(self, data: torch.Tensor, targets: torch.Tensor = None) -> None:
        self.data = data.detach().clone()
        if targets:
            self.targets = targets.detach().clone()
            self._has_targets = True

    def append(self, data: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        if self.data.numel() == 0:
            self.data = data.detach().clone()
        else:
            tmp_data = data.detach().clone().to(self._device)
            if self.data.dim() > tmp_data.dim():
                tmp_data = tmp_data.unsqueeze(0)
            elif self.data.dim() < tmp_data.dim():
                raise ValueError("Wrong data sizes!")
            self.data = torch.cat([self.data, tmp_data], dim=0)

        if targets is not None:
            if self.targets.numel() == 0:
                self.targets = targets.detach().clone()
            else:
                self.targets = torch.cat([self.targets, targets.detach().clone()], dim=0)
            self._has_targets = True

    def remove(self, idxs):
        mask = torch.ones((self.data.size(0),), dtype=bool)
        mask[idxs] = False
        self.data = self.data[mask]
        if self._has_targets:
            self.targets = self.targets[mask]

    def set_transforms(self, transforms: Transforms):
        self._transforms = transforms
        self._vectorized_transforms = torch.vmap(transforms)

    def sizes(self) -> List[int]:
        return list(self.data.size())

    def __len__(self) -> int:
        return self.data.size(0)

    def numel(self) -> int:
        return self.data.numel()

    def data_ref(self) -> torch.Tensor:
        return self.data

    def targets_ref(self) -> torch.Tensor:
        return self.targets

    def _split(self, percentage: float):
        split_1_targets = None
        split_2_targets = None
        indices = torch.randperm(self.data.size(0))
        num_el_after_split = int(percentage * self.data.size(0))

        split_2_data = self.data[indices[num_el_after_split:]]
        split_1_data = self.data[indices[:num_el_after_split]]

        if self._has_targets:
            split_1_targets = self.targets[indices[:num_el_after_split]]
            split_2_targets = self.targets[indices[num_el_after_split:]]
        return (split_1_data, split_1_targets), (split_2_data, split_2_targets)
