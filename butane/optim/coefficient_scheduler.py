import torch


class CoefficientScheduler:

    def __init__(
        self,
        start: float,
        end: float,
        max_steps: int,
        mode: str = 'linear'
    ):
        self._start = start
        self._end = end
        self._mode = mode

        self._max_steps = max_steps

        if mode == 'linear':
            self._coeff = torch.linspace(self._start, self._end, self._max_steps)
        elif mode == 'constant':
            self._coeff = torch.full((self._max_steps,), self._start)
        elif mode == 'cosine':
            _t = torch.linspace(0, 1, self._max_steps)
            self._coeff = self._end + 0.5 * (self._start - self._end) * (1 + torch.cos(torch.pi * _t))

        self.index = 0

    def update(self):
        self.index += 1

    def __call__(self):
        return self._coeff[self.index].item()


