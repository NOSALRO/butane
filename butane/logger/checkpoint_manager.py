import torch
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

from .._typedefs import ModuleParams
from .. import nn

class CheckpointManager:

    def __init__(self, fpath: Path, logger):
        self.fpath = Path(fpath)
        self.logger = logger

    def save(
        self,
        step: int,
        monitor_state: Optional[dict] = None,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        ema: Optional[Any] = None,
        **modules,
    ) -> Tuple[Path, Path]:
        self._validate_types(model, optimizer, lr_scheduler, ema)

        _path = self.fpath / f"checkpoint_{step}"
        output_path = _path / "outputs"
        output_path.mkdir(parents=True, exist_ok=True)

        cp = dict(
            step=step,
            **self._create_dict(model, "model"),
            **self._create_dict(optimizer, "optimizer"),
            **self._create_dict(lr_scheduler, "lr_scheduler"),
            **self._create_dict(ema, "ema"),
        )
        for k, v in modules.items():
            cp.update(self._create_dict(v, k))

        if monitor_state:
            cp['monitor_state'] = monitor_state

        torch.save(cp, _path / "checkpoint.pt")
        self.logger.info(f"Checkpoint saved: checkpoint_{step}")

        return _path, output_path

    def load(
        self,
        step: int,
        model: Any,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        ema: Optional[Any] = None,
        **modules,
    ) -> Tuple[dict, Path, Path]:
        """Loads weights from disk and returns (checkpoint_dict, checkpoint_path, output_path)."""
        ckpt_folder = self.fpath.absolute() / f"checkpoint_{step}"

        if not ckpt_folder.exists():
             self.logger.error(f"Checkpoint not found at: {ckpt_folder}")
             raise FileNotFoundError(f"Checkpoint {step} not found.")

        checkpoint = nn.utils.load_state(
            str(ckpt_folder),
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            **modules,
        )

        output_path = ckpt_folder / "outputs"
        return checkpoint, ckpt_folder, output_path

    @staticmethod
    def _validate_types(model, optimizer, lr_scheduler, ema):
        is_mod = lambda o: isinstance(o, torch.nn.Module)
        is_mod_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.nn.Module) for x in o)
        is_opt = lambda o: isinstance(o, torch.optim.Optimizer)
        is_opt_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.optim.Optimizer) for x in o)
        is_sched = lambda o: isinstance(o, torch.optim.lr_scheduler.LRScheduler)
        is_sched_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.optim.lr_scheduler.LRScheduler) for x in o)

        if model is not None: assert is_mod(model) or is_mod_list(model), "`model` type error."
        if optimizer is not None: assert is_opt(optimizer) or is_opt_list(optimizer), "`optimizer` type error."
        if lr_scheduler is not None: assert is_sched(lr_scheduler) or is_sched_list(lr_scheduler), "`lr_scheduler` type error."
        if ema is not None: assert is_mod(ema) or is_mod_list(ema), "`ema` type error."

    @staticmethod
    def _create_dict(obj: object, key: str) -> Dict[str, Any]:
        if obj is None: return {key: None}
        if not isinstance(obj, (list, tuple)):
            assert hasattr(obj, "state_dict"), f"Object {key} does not have state_dict()"
            return {key: obj.state_dict()}

        out_dict = {}
        for i, o in enumerate(obj, start=1):
            assert hasattr(o, "state_dict"), f"Object {key}_{i} does not have state_dict()"
            out_dict.update({f'{key}_{i}': o.state_dict()})
        return out_dict
