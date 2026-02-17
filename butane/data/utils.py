from typing import Optional, Union, Iterator
import torch

@torch.jit.script
def batch_arange(
    starts: torch.Tensor,
    stops: Optional[torch.Tensor] = None,
    steps: Optional[Union[torch.Tensor, int]] = None
) -> torch.Tensor:

    if starts is None and stops is None:
        raise TypeError

    if stops is None and starts is not None:
        stops = starts
        starts = torch.zeros_like(stops)

    if starts is None:
        starts = torch.zeros_like(stops)

    if stops is None:
        stops = torch.zeros_like(starts)

    if isinstance(steps, int):
        steps = torch.full_like(starts, steps, dtype=torch.int64)

    if steps is None:
        steps = torch.ones_like(starts, dtype=torch.int64)

    if len(torch.unique(starts - stops)) != 1:
        raise TypeError

    steps = steps.to(dtype=torch.int64)

    aranges = []
    for start, stop, step in zip(starts, stops, steps):
        aranges.append(torch.arange(start, stop, step, dtype=torch.int64).unsqueeze(0))

    return torch.vstack(aranges)

def InfiniteIterator(dataloader: torch.utils.data.DataLoader) -> Iterator:
    while True:
        for sample in dataloader:
            yield sample
