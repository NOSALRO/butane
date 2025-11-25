from typing import Optional, List, Tuple, Dict, Union
import torch

from ..transforms import Transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        *,
        on_demand_device_load: bool = False,
        return_tuple: bool = False,
        device: torch.device = 'cpu',
    ) -> None:
        super().__init__()
        self.data = torch.tensor([]) if data is None else data.detach().clone().float()
        self.targets = torch.tensor([]) if targets is None else targets.detach().clone()
        self._device = device
        self._transforms, self._vectorized_transforms = None, None
        self._on_demand_device_load = on_demand_device_load
        self._return_tuple = return_tuple
        if not self._on_demand_device_load:
            self.data = self.data.to(self._device)
            self.targets = self.targets.to(self._device)

    def _has_targets(self):
        return self.targets is not None and self.targets.numel() > 0

    def __getitem__(self, idx: Union[int, List[int]]) -> Dict[str, torch.Tensor]:
        _data = self.data[idx]
        if self._on_demand_device_load:
            _data = _data.to(self._device)

        if isinstance(idx, int):
            _data = _data if self._transforms is None else self._transforms(_data)
        else:
            _data = _data if self._transforms is None else self._vectorized_transforms(_data)

        if self._has_targets():
            _targets = self.targets[idx]
            _targets = _targets.to(_data.device)
            return self._convert_to_tuple({"data": _data, "targets": _targets})
        else:
            return self._convert_to_tuple({ "data": _data })

    def split(self, percentage: float):
        _transforms = self._transforms
        (split_1_data, split_1_targets), (split_2_data, split_2_targets) = self._split(percentage)
        self.__init__(
            data=split_1_data,
            targets=split_1_targets,
            on_demand_device_load=self._on_demand_device_load,
            device=self._device,
            return_tuple=self._return_tuple
        )
        self.set_transforms(_transforms)
        splitted_ds = Dataset(
            data=split_2_data,
            targets=split_2_targets,
            on_demand_device_load=self._on_demand_device_load,
            device=self._device,
            return_tuple=self._return_tuple)
        splitted_ds.set_transforms(_transforms)
        return splitted_ds

    def flatten(self, dim: int = 1) -> None:
        self.data = self.data.flatten(start_dim=dim)

    def to(self, device: torch.device) -> None:
        if not self._on_demand_device_load:
            self.data = self.data.to(device)
            if self._has_targets():
                self.targets = self.targets.to(device)
        self._device = device
        return self

    def save(self, filepath: str) -> None:
        base_path = filepath.rsplit('.', 1)[0]
        torch.save(self.data.cpu().detach(), f"{base_path}_data.pt")
        if self._has_targets():
            torch.save(self.targets.cpu().detach(), f"{base_path}_targets.pt")

    def set(self, data: torch.Tensor, targets: torch.Tensor = None) -> None:
        self.data = data.detach().clone()
        if targets is not None:
            self.targets = targets.detach().clone()

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

    def remove(self, idxs):
        mask = torch.ones((self.data.size(0),), dtype=bool)
        mask[idxs] = False
        self.data = self.data[mask]
        if self._has_targets():
            self.targets = self.targets[mask]

    def set_transforms(self, transforms: Transforms):
        self._transforms = transforms
        self._vectorized_transforms = torch.vmap(transforms)

    def transforms(self) -> Transforms:
        return self._transforms

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

    def return_tuple(self) -> None:
        self._return_tuple = True

    def _split(self, percentage: float):
        split_1_targets = None
        split_2_targets = None
        indices = torch.randperm(self.data.size(0))
        num_el_after_split = int(percentage * self.data.size(0))

        split_2_data = self.data[indices[num_el_after_split:]]
        split_1_data = self.data[indices[:num_el_after_split]]

        if self._has_targets():
            split_1_targets = self.targets[indices[:num_el_after_split]]
            split_2_targets = self.targets[indices[num_el_after_split:]]
        return (split_1_data, split_1_targets), (split_2_data, split_2_targets)

    def _convert_to_tuple(self, returns: dict) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        if self._return_tuple:
            data = returns['data']
            if self._has_targets():
                target = returns['targets']
                return data, target
            return data
        else:
            return returns
