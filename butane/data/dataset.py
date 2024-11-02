from typing import Optional, List, Tuple, Dict
import torch

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._has_targets:
            return {
                "data": self.data[idx],
                "targets": self.targets[idx]
            }
        else:
            return { "data": self.data[idx] }

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

class TrajectoryDataset(Dataset):

    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        horizon: int = 16,
        loop: bool = False
    ) -> None:
        super().__init__()

        self.data = torch.tensor([]) if data is None else data.detach().clone().float()
        self.horizon = horizon
        self._loop = loop
        self.num_trajectories, self.num_steps, _ = self.data.shape
        self._device = torch.device('cpu')
        self._is_data_prepared = False

    def __len__(self) -> int:
        return self.num_trajectories * self.num_steps

    def __getitem__(self, idx: int) -> torch.Tensor:

        if not self._is_data_prepared:
            self.__prepare_data()

        traj_index = idx // self.num_steps
        step_index = idx % self.num_steps
        seq_current = self.data[traj_index, step_index: step_index + self.horizon]
        seq_future = self.data[traj_index, (step_index + self.horizon): step_index + 2*self.horizon]
        return {"data": seq_current, "target": seq_future}

    def __prepare_data(self):
        if not self._loop:
            self.data = torch.cat([self.data, self.data[:, [-1], :].repeat(1, 2*self.horizon, 1)], dim=1)
        else:
            self.data = torch.cat([self.data, self.data[:, :self.horizon, :].repeat(1, 2, 1)], dim=1)
        self._is_data_prepared = True
