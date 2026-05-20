import copy
from typing import Any

import numpy as np
import torch

from ..utils import batch_arange
from .dataset import Dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor | np.ndarray] | None = None,
        *,
        horizon: int = 1,
        history: int = 1,
        align_start: bool = False,
        on_demand_device_load: bool = False,
        return_tuple: bool = False,
        device: torch.device = "cpu",
    ) -> None:

        super().__init__(
            on_demand_device_load=on_demand_device_load,
            return_tuple=return_tuple,
            device=device,
        )

        assert horizon > 0, "Horizon cannot be less than 1."
        assert history > 0, "History cannot be less than 1."

        self.episode_lens: torch.Tensor | None = None
        self.episode_start: torch.Tensor | None = None
        self.episode_end: torch.Tensor | None = None

        self._ingest_data(data)
        self.horizon = horizon
        self.history = history - 1
        self.align_start = align_start

        indices = torch.arange(len(self.sample_to_episode), dtype=torch.long)
        ep_starts = self.episode_start[self.sample_to_episode]
        ep_ends = self.episode_end[self.sample_to_episode]

        sample_start = torch.maximum(indices - self.history, ep_starts)
        if not self.align_start:
            sample_ends = torch.minimum(indices + self.horizon, ep_ends)
        else:
            sample_ends = torch.minimum(sample_start + self.horizon, ep_ends)

        self.samples = torch.stack([self.sample_to_episode, sample_start, sample_ends], dim=1)

        self.pad_left_fn = self._default_pad_left
        self.pad_right_fn = self._default_pad_right

    def _sample(
        self, tensor: torch.Tensor, idx: int | list[Any] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Extract integer values for slicing
        start_idx = sample[1].item()
        end_idx = sample[2].item()

        pad_left_len = start_idx - (idx - self.history)
        history_seq = tensor[start_idx : idx + 1]

        if pad_left_len > 0:
            history_seq = self.pad_left_fn(history_seq, pad_left_len)

        if self.align_start:
            pad_right_len = start_idx + self.horizon - end_idx
            horizon_seq = tensor[start_idx:end_idx]
        else:
            pad_right_len = idx + self.horizon - end_idx
            horizon_seq = tensor[idx:end_idx]

        if pad_right_len > 0:
            horizon_seq = self.pad_right_fn(horizon_seq, pad_right_len)

        return history_seq, horizon_seq

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
        if isinstance(self.data, torch.Tensor):
            return self._sample(self.data, idx)

        data_dict = {}
        target_dict = {}

        for key, tensor in self.data.items():
            hist, horiz = self._sample(tensor, idx)
            data_dict[key] = hist
            target_dict[key] = horiz

        return {
            "data": data_dict,
            "targets": target_dict
        }

    def __len__(self) -> int:
        return len(self.sample_to_episode)

    def _ingest_data(
        self,
        data: torch.Tensor
        | np.ndarray
        | list[torch.Tensor | np.ndarray]
        | dict[str, torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray]],
    ):
        if isinstance(data, (torch.Tensor, np.ndarray)):
            data_tensor = torch.as_tensor(data)
            n_episodes, n_steps = data_tensor.shape[0], data_tensor.shape[1]
            self.episode_lens = torch.full(size=(n_episodes,), fill_value=n_steps, dtype=torch.long)
            self.sample_to_episode = torch.arange(len(self.episode_lens)).repeat_interleave(n_steps)
            self.data = data_tensor.reshape(n_episodes * n_steps, *data_tensor.shape[2:])

        elif isinstance(data, list):
            n_episodes = len(data)
            self.episode_lens = torch.tensor([len(d) for d in data], dtype=torch.long)
            self.sample_to_episode = torch.cat(
                [torch.full(size=(self.episode_lens[i],), fill_value=i) for i in range(n_episodes)],
                dim=-1,
            )
            self.data = torch.cat([torch.as_tensor(d) for d in data], dim=0)

        elif isinstance(data, dict):
            self.data = {}
            first_val = next(iter(data.values()))

            if isinstance(first_val, (torch.Tensor, np.ndarray)):
                n_episodes, n_steps = first_val.shape[0], first_val.shape[1]
                self.episode_lens = torch.full(
                    size=(n_episodes,), fill_value=n_steps, dtype=torch.long
                )
                self.sample_to_episode = torch.arange(len(self.episode_lens)).repeat_interleave(
                    n_steps
                )
            elif isinstance(first_val, list):
                n_episodes = len(first_val)
                self.episode_lens = torch.tensor([len(d) for d in first_val], dtype=torch.long)
                self.sample_to_episode = torch.cat(
                    [
                        torch.full(size=(self.episode_lens[i],), fill_value=i)
                        for i in range(n_episodes)
                    ],
                    dim=-1,
                )

            for key, val in data.items():
                val_tensor = torch.as_tensor(val) if not isinstance(val, list) else val
                if isinstance(val_tensor, torch.Tensor):
                    self.data[key] = val_tensor.reshape(-1, *val_tensor.shape[2:])
                elif isinstance(val, list):
                    self.data[key] = torch.cat([torch.as_tensor(d) for d in val], dim=0)

        self.episode_end = torch.cumsum(self.episode_lens, dim=-1)
        self.episode_start = torch.cat([torch.tensor([0], dtype=torch.long), self.episode_end[:-1]])

    def _prepare_idx(
        self, idx: int | list[int] | tuple[int] | slice | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(idx, int):
            return torch.tensor([idx]).long()
        elif isinstance(idx, (list, tuple)):
            return torch.tensor(idx).long()
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.data.size(0))
            return torch.arange(start, stop, step, dtype=torch.long)
        elif isinstance(idx, torch.Tensor):
            return idx.long()

    @staticmethod
    def _default_pad_left(input: torch.Tensor, pad_len: torch.Tensor) -> torch.Tensor:
        return torch.cat([input[0:1].repeat_interleave(pad_len, dim=0), input], dim=0)
        # padded = [
        #     torch.cat([x[0:1].repeat_interleave(n, dim=0), x], dim=0)
        #     for x, n in zip([input], [pad_len])
        # ]
        # return torch.stack(padded, dim=0)

    @staticmethod
    def _default_pad_right(input: torch.Tensor, pad_len: torch.Tensor) -> torch.Tensor:
        return torch.cat([input, input[-1:].repeat_interleave(pad_len, dim=0)], dim=0)
        # padded = [
        #     torch.cat([x, x[-1:].repeat_interleave(n, dim=0)], dim=0)
        #     for x, n in zip(input, pad_len)
        # ]
        # return torch.stack(padded, dim=0)
