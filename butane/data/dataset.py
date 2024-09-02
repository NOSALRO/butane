import torch
from typing import Optional, List, Tuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None):
        super().__init__()
        self.data = torch.tensor([]) if data is None else data.detach().clone().float()
        self.targets = torch.tensor([]) if targets is None else targets.detach().clone()
        self._has_targets = targets is not None
        self._device = torch.device('cpu')

    def __getitem__(self, idx):
        if self._has_targets:
            return {
                "data": self.data[idx],
                "targets": self.targets[idx]
            }
        else:
            return { "data": self.data[idx] }

    def flatten(self, dim: int = 1):
        self.data = self.data.flatten(start_dim=dim)

    def to(self, device: torch.device):
        self.data = self.data.to(device)
        if self._has_targets:
            self.targets = self.targets.to(device)
        self._device = device

    def save(self, filepath: str):
        base_path = filepath.rsplit('.', 1)[0]
        torch.save(self.data.cpu().detach(), f"{base_path}_data.pt")
        if self._has_targets:
            torch.save(self.targets.cpu().detach(), f"{base_path}_targets.pt")

    def set(self, data: torch.Tensor, targets: torch.Tensor = None):
        self.data = data.detach().clone()
        if targets:
            self.targets = targets.detach().clone()
            self._has_targets = True

    def append(self, data: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if self.data.numel() == 0:
            self.data = data.detach().clone()
        else:
            tmp_data = data.detach().clone()
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