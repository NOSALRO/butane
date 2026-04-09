import copy
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..utils import batch_arange
from .dataset import Dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data: Optional[torch.Tensor],
        horizon: int,
        *,
        context: Optional[int] = None,
        shift: int = 1,
        loop: bool = False,
        drop_last: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        assert shift > 0, "shift cannot be 0"

        self.data = torch.tensor([]) if data is None else data.detach().clone().float()
        self.horizon = horizon
        self.shift = shift
        self._loop = loop
        self._drop_last = drop_last
        self.context = context if context is not None else horizon
        _, self.num_steps, self.feature_dims = self.data.shape

        self.__prepare_data()
        if not self._on_demand_device_load:
            self.data = self.data.to(self._device)
            self.targets = self.targets.to(self._device)

        if self._drop_last and self.num_steps < 2 * self.horizon:
            raise ValueError("time series too short for the requested horizon")

    def __len__(self) -> int:
        if not self._drop_last:
            return ((self.data.size(0) * (self.num_steps - self.context)) // self.shift) + 1
        else:
            return (
                (self.data.size(0) * (self.num_steps - (self.context + self.horizon))) // self.shift
            ) + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not isinstance(idx, int):
            return self.__batched_get(idx)

        idx *= self.shift
        if not self._drop_last:
            traj_index = idx // self.num_steps
            step_index = idx % self.num_steps
        else:
            traj_index = idx // (self.num_steps - (self.context + self.horizon - 1))
            step_index = idx % (self.num_steps - (self.context + self.horizon - 1))

        end_current = step_index + self.context
        end_future = end_current + self.horizon

        seq_current = self.data[traj_index, step_index:end_current]
        seq_future = self.data[traj_index, end_current:end_future]
        if self._on_demand_device_load:
            seq_current = seq_current.to(self._device)
            seq_future = seq_future.to(self._device)
        return self._convert_to_tuple(
            {
                "data": seq_current if self._transforms is None else self._transforms(seq_current),
                "targets": seq_future if self._transforms is None else self._transforms(seq_future),
            }
        )

    def split(self, percentage: float, generator: Optional[torch.Generator] = None):
        if percentage == 0:
            return None
        (split_1_data, _), (split_2_data, _) = self._split(percentage, generator=generator)
        transforms = self._transforms
        self.__init__(
            data=split_1_data,
            horizon=self.horizon,
            context=self.context,
            drop_last=self._drop_last,
            shift=self.shift,
            loop=self._loop,
            on_demand_device_load=self._on_demand_device_load,
            device=self._device,
            return_tuple=self._return_tuple,
        )
        self.set_transforms(transforms)
        splitted = TrajectoryDataset(
            data=split_2_data,
            horizon=self.horizon,
            context=self.context,
            drop_last=self._drop_last,
            shift=self.shift,
            loop=self._loop,
            on_demand_device_load=self._on_demand_device_load,
            device=self._device,
            return_tuple=self._return_tuple,
        )
        splitted.set_transforms(self._transforms)
        return splitted

    def convert_to_dataset(self):
        # transforms = self._transforms
        instances = self[:]
        dataset = Dataset(instances["data"], instances["targets"])
        # dataset.set_transforms(transforms)
        return dataset

    def size(self, dim: Optional[int] = None) -> Union[List[int], int]:
        return list(self[:]["data"].size(dim)) if dim is None else self[:]["data"].size(dim)

    def __prepare_data(self):
        if not (self._loop or self._drop_last):
            self.data = torch.cat(
                [self.data, self.data[:, [-1], :].repeat(1, (self.context + self.horizon), 1)],
                dim=1,
            )
        elif self._drop_last:
            ...
        else:
            if self.horizon * 2 >= self.num_steps:
                self.data = torch.cat(
                    [self.data, self.data[:, : self.horizon, :].repeat(1, 2, 1)], dim=1
                )
            else:
                self.data = torch.cat(
                    [self.data, self.data[:, : (self.context + self.horizon), :]], dim=1
                )

    def __batched_get(self, idxs: Union[slice, list]):
        idx_list = None
        if isinstance(idxs, slice):
            start = idxs.start
            start = 0 if start is None else start

            stop = idxs.stop
            stop = len(self) if stop is None else stop

            step = idxs.step
            step = 1 if step is None else step
            start, stop, step = start * self.shift, stop * self.shift, step * self.shift
            idx_list = torch.arange(start, stop, step)
        else:
            idx_list = torch.tensor(idxs) * self.shift

        if not self._drop_last:
            traj_idx_list = (idx_list // self.num_steps).unsqueeze(-1)
            step_idx_list = idx_list % self.num_steps
        else:
            traj_idx_list = (
                idx_list // (self.num_steps - (self.context + self.horizon - 1))
            ).unsqueeze(-1)
            step_idx_list = idx_list % (self.num_steps - (self.context + self.horizon - 1))

        step_idx_list_data = batch_arange(step_idx_list, step_idx_list + self.context)
        step_idx_list_targets = batch_arange(
            step_idx_list + self.context, step_idx_list + (self.context + self.horizon)
        )

        seq_current = self.data[traj_idx_list, step_idx_list_data]
        seq_future = self.data[traj_idx_list, step_idx_list_targets]
        if self._on_demand_device_load:
            seq_current = seq_current.to(self._device)
            seq_future = seq_future.to(self._device)

        return self._convert_to_tuple(
            {
                "data": (
                    seq_current
                    if self._transforms is None
                    else self._vectorized_transforms(seq_current)
                ),
                "targets": (
                    seq_future
                    if self._transforms is None
                    else self._vectorized_transforms(seq_future)
                ),
            }
        )
