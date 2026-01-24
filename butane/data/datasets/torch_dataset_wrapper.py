import torch


class TorchDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, torch_dataset: torch.utils.data.Dataset):
        super().__init__()
        self.ds = torch_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        out = self.ds[idx]
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return dict(data=out[0], targets=out[1])
        elif isinstance(out, (tuple, list)) and len(out) == 1:
            return dict(data=out[0], targets=None)
        elif isinstance(out, torch.Tensor):
            return dict(data=out, targets=None)

