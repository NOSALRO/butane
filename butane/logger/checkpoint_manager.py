import json
import shutil
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, Union

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
        is_best: bool = False,
        **modules,
    ) -> Tuple[Path, Path]:
        self._validate_types(
            model=modules.get("model"),
            optimizer=modules.get("optimizer"),
            lr_scheduler=modules.get("lr_scheduler"),
            ema=modules.get("ema"),
        )

        _path = self.fpath / f"checkpoint_{step}"
        output_path = _path / "outputs"
        output_path.mkdir(parents=True, exist_ok=True)

        cp = dict(step=step)
        for k, v in modules.items():
            if v is not None: cp.update(self._create_dict(v, k))

        if monitor_state:
            cp['monitor_state'] = monitor_state

        torch.save(cp, _path / "checkpoint.pt")
        self.logger.info(f"Checkpoint saved: checkpoint_{step}")

        if is_best:
            best_dir = self.fpath / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)

            best_path = best_dir / "checkpoint.pt"
            shutil.copyfile(_path / "checkpoint.pt", best_path)

            if monitor_state and 'best_metrics' in monitor_state:
                ledger_path = best_dir / "best_metrics.jsonl"
                with open(ledger_path, "a", encoding="utf-8") as f:
                    entry = {"step": step, **monitor_state['best_metrics']}
                    f.write(json.dumps(entry) + "\n")

        return _path, output_path

    def load(
        self,
        step: Union[int, str],
        **modules,
    ) -> Tuple[dict, Path, Path]:

        if str(step).lower() == "best":
            ckpt_folder = self.fpath.absolute() / f"best_model/"
            if not ckpt_folder.exists():
                self.logger.error(f"No best checkpoint found at {ckpt_folder}")
                raise FileNotFoundError(f"Best checkpoint missing.")
            self.logger.info("Loading the absolute best model weights...")
        else:
            potential_path = Path(step)
            if potential_path.exists() and potential_path.is_dir():
                ckpt_folder = potential_path

                # Update the root fpath to the parent of this explicit checkpoint
                self.fpath = ckpt_folder.parent
                try:
                    parsed_step = ckpt_folder.name.split('_')[-1]
                    self.logger.info(f"Loaded explicit path. Inferred step {parsed_step}. Root updated to {self.fpath}")
                except Exception:
                    self.logger.warning(f"Could not parse step from folder name: {ckpt_folder.name}")
            else:
                ckpt_folder = self.fpath.absolute() / f"checkpoint_{step}"

        checkpoint = nn.utils.load_state(str(ckpt_folder), **modules)

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
