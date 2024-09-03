from typing import Optional, Callable, Union, Tuple
import torch

class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dl: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ) -> None:
        self.model = model
        self.dl = dl
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(
        self,
        epochs: int,
        loss_fn: Optional[Callable[[torch.Tensor, ...], Tuple[torch.Tensor,...]]],
        eval_period: Optional[int] = 0,
        eval_dl: Optional[torch.utils.data.DataLoader] = None
    ) -> None:
        for epoch in range(epochs):
            print(f"Epoch {epoch} -> ", end='')
            self.model.step(self.dl, self.optimizer, loss_fn, self.scheduler)
            if eval_period and eval_dl and not ((epoch + 1) % eval_period):
                eval(eval_dl.value(), loss)

    def eval(eval_dl: torch.utils.data.DataLoader, loss_fn: Optional[Callable[[torch.Tensor, ...], Tuple[torch.Tensor,...]]]) -> None:
        self.model.eval()
        print("Evaluation -> ", end='')
        self.model.step(eval_dl, self.optimizer, loss_fn, self.scheduler)
        self.model.train()
