from typing import Optional, List, Tuple, Dict, Union
import torch

from .dataset import Dataset
from .._utils import batch_arange


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
        _, self.num_steps, _ = self.data.shape
        self._device = torch.device('cpu')
        self._is_data_prepared = False

    def __len__(self) -> int:
        return self.data.size(0) * self.num_steps

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not self._is_data_prepared:
            self.__prepare_data()

        if not isinstance(idx, int):
            return self.__batched_get(idx)

        traj_index = idx // self.num_steps
        step_index = idx % self.num_steps
        seq_current = self.data[traj_index, step_index: step_index + self.horizon]
        seq_future = self.data[traj_index, (step_index + self.horizon): step_index + 2*self.horizon]
        return {"data": seq_current if self._transforms is None else self._transforms(seq_current),
                "targets": seq_future if self._transforms is None else self._transforms(seq_future)}

    def split(self, percentage: float):
        (split_1_data, _), (split_2_data, _) = self._split(percentage)
        self.__init__(split_1_data, horizon=self.horizon, loop=self._loop)
        return TrajectoryDataset(split_2_data, horizon=self.horizon, loop=self._loop)

    def __prepare_data(self):
        if not self._loop:
            self.data = torch.cat([self.data, self.data[:, [-1], :].repeat(1, 2*self.horizon, 1)], dim=1)
        else:
            if self.horizon * 2 >= self.num_steps:
                self.data = torch.cat([self.data, self.data[:, :self.horizon, :].repeat(1, 2, 1)], dim=1)
            else:
                self.data = torch.cat([self.data, self.data[:, :2*self.horizon, :]], dim=1)
        self._is_data_prepared = True

    def __batched_get(self, idxs: Union[slice, list]):
        idx_list = None
        if isinstance(idxs, slice):
            start = idxs.start
            start = 0 if start is None else start

            stop = idxs.stop
            stop = len(self) if stop is None else stop

            step = idxs.step
            step = 1 if step is None else step
            idx_list = torch.arange(start, stop, step)
        else:
            idx_list = torch.tensor(idxs)

        traj_idx_list = (idx_list // self.num_steps).unsqueeze(-1)
        step_idx_list = idx_list % self.num_steps
        step_idx_list_data = batch_arange(step_idx_list, step_idx_list + self.horizon)
        step_idx_list_targets = batch_arange(step_idx_list + self.horizon, step_idx_list + 2*self.horizon)

        return {
            "data": self.data[traj_idx_list, step_idx_list_data] if self._transforms is None
                else self._vectorized_transforms(self.data[traj_idx_list, step_idx_list_data]),
            "targets": self.data[traj_idx_list, step_idx_list_targets] if self._transforms is None
                    else self._vectorized_transforms(self.data[traj_idx_list, step_idx_list_targets])
        }
