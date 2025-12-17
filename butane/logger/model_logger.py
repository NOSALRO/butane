import os
import sys
import csv
import yaml
import torch
import numbers
import shutil
import logging
import datetime
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

from .._typedefs import ModuleParams
from .. import nn

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


class _LiteralDumper(yaml.SafeDumper):
    pass

def _multiline_str_presenter(dumper: yaml.SafeDumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)
_LiteralDumper.add_representer(str, _multiline_str_presenter)

class _ModelMonitor:

    def __init__(
        self,
        logger: logging.Logger,
        *,
        increase_keys: Optional[List[str]] = None,
        decrease_keys: Optional[List[str]] = None,
        tolerance: float = 0.1,
    ):
        self.logger = logger
        self._increase_keys = increase_keys if increase_keys is not None else []
        self._decrease_keys = decrease_keys if decrease_keys is not None else []
        self._tolerance = tolerance

        # State
        self.best_step = -1
        self.best_metrics = {}

    def __call__(self, step: int, status: dict) -> bool: 
        if self.best_step < 0:
            self.best_step = step
            self.best_metrics = status
            return False

        degraded = self._check_degradation(status)

        if degraded:
            self.logger.warning(f"[Monitor] Step {step}: Metrics degraded > {self._tolerance*100:.0f}% vs Step {self.best_step}")
            return True
        else:
            self.best_step = step
            self.best_metrics = status
            self.logger.info(f"[Monitor] Step {step}: Metrics OK")
            return False

    def _check_degradation(self, current: Dict[str, float]) -> bool:
        for ik in self._increase_keys:
            if ik not in current or ik not in self.best_metrics: continue
            best, new = self.best_metrics[ik], current[ik]
            if new < best * (1 - self._tolerance):
                return True

        for dk in self._decrease_keys:
            if dk not in current or dk not in self.best_metrics: continue
            best, new = self.best_metrics[dk], current[dk]
            if new > best * (1 + self._tolerance):
                return True

        return False

    def state(self):
        return {
            'best_step': self.best_step,
            'best_metrics': self.best_metrics
        }

    def load_state(self, state):
        self.best_step = state.get('best_step', -1)
        self.best_metrics = state.get('best_metrics', {})

class ModelLogger:

    def __init__(
        self,
        fpath: str,
        overwrite: bool = False,
        resume: bool = False,
        use_wandb: bool = False,
    ):
        self.fpath = Path(fpath)
        self._overwrite = overwrite

        if self.fpath.exists() and not self._overwrite and not resume:
            ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            self.fpath = self.fpath.with_name(f"{self.fpath.name}_{ts}")
        self.fpath.mkdir(parents=True, exist_ok=True)


        self._use_rollback = False
        self._rollback_monitor = None

        self.logger = self._setup_logger()
        self.logger.info(f"Initialized Experiment at: {self.fpath}")

        self._stats = {}
        self._config = {}
        self._last_used_path, self._output_path = None, None
        self._step = 0

        self._use_wandb = _HAS_WANDB and use_wandb
        if self._use_wandb:
            project = os.environ.get("WANDB_PROJECT")
            assert project is not None, "Set the WANDB_PROJECT env variable"

            id_file = self.fpath / "wandb_id.txt"
            self._wandb_defined_metrics = set()
            if id_file.exists() and resume:
                run_id = id_file.read_text().strip()
                run_name = self.fpath.name
                self.logger.info(f"Resuming existing WandB Run ID: {run_id}")
            else:
                run_id = wandb.util.generate_id()
                run_name = f"{self.fpath.name}_{run_id}"
                id_file.write_text(run_id)
                self.logger.info(f"Created new WandB Run ID: {run_id}")

            wandb.init(
                project=project,
                name=run_name,   # Display Name
                id=run_id,       # Internal Unique ID
                dir=str(self.fpath),
                resume="allow"
            )

    def checkpoint(
        self,
        step: int,
        *,
        model: Optional[ModuleParams] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        ema: ModuleParams = None,
        scaler: Optional[torch.nn.Module] = None,
    ):

        assert isinstance(step, int), f"'step' must be int, got {type(step).__name__}"
        self._step = step
        is_mod = lambda o: isinstance(o, torch.nn.Module)
        is_mod_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.nn.Module) for x in o)
        is_opt = lambda o: isinstance(o, torch.optim.Optimizer)
        is_opt_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.optim.Optimizer) for x in o)
        is_sched = lambda o: isinstance(o, torch.optim.lr_scheduler.LRScheduler)
        is_sched_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.optim.lr_scheduler.LRScheduler) for x in o)

        if model is not None:
            assert is_mod(model) or is_mod_list(model), "`model` must be a torch.nn.Module or a list/tuple of torch.nn.Modules."
        if optimizer is not None:
            assert is_opt(optimizer) or is_opt_list(optimizer), "`optimizer` must be a torch.optim.Optimizer or list/tuple of them."
        if lr_scheduler is not None:
            assert is_sched(lr_scheduler) or is_sched_list(lr_scheduler), "`lr_scheduler` must be a torch.optim.lr_scheduler.LRScheduler or list/tuple of them."
        if ema is not None:
            assert is_mod(ema) or is_mod_list(ema), "`ema` must be a torch.nn.Module or list/tuple of torch.nn.Modules."
        if scaler is not None:
            assert isinstance(scaler, torch.nn.Module), "`scaler` must be a torch.nn.Module or None."

        _path = self.fpath / f"checkpoint_{step}"
        output_path = _path / "outputs/"
        output_path.mkdir(parents=True, exist_ok=True)

        cp = dict(
            step=self._step,
            **self._create_dict(model, "model"),
            **self._create_dict(optimizer, "optimizer"),
            **self._create_dict(lr_scheduler, "lr_scheduler"),
            **self._create_dict(ema, "ema"),
            **self._create_dict(scaler, "scaler"),
        )

        if getattr(self, '_use_rollback', False) and self._rollback_monitor:
            cp['monitor_state'] = self._rollback_monitor.state()

        torch.save(cp, _path / "checkpoint.pt")
        self.save_stats(self.fpath)

        self._last_used_path = _path
        self._output_path = str(output_path)
        self._step = step
        self.logger.info(f"Checkpoint saved: checkpoint_{step}")

    def load_checkpoint(
        self,
        step: int,
        *,
        model: ModuleParams,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        ema: ModuleParams = None,
        scaler: Optional[torch.nn.Module] = None,
    ):
        ckpt_folder = self.fpath.absolute() / f"checkpoint_{step}"

        if not ckpt_folder.exists():
             self.logger.error(f"Checkpoint not found at: {ckpt_folder}")
             raise FileNotFoundError(f"Checkpoint {step} not found.")

        checkpoint = nn.utils.load_state(
            str(ckpt_folder),
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema, scaler=scaler
        )

        loaded_step = checkpoint.get("step")
        self._step = loaded_step if loaded_step is not None else 0

        if getattr(self, '_use_rollback', False) and self._rollback_monitor:
            monitor_state = checkpoint.get('monitor_state')
            if monitor_state:
                self._rollback_monitor.load_state_dict(monitor_state)
                self.logger.info(f"Rollback Monitor state restored (Best Step: {self._rollback_monitor.best_step})")

        if ckpt_folder.exists():
            self._last_used_path = ckpt_folder
            self._output_path = str(ckpt_folder / "outputs")
            self.logger.info(f"State restored from step {self._step}. Output path set.")

    def enable_rollback(
        self,
        increase_keys: List[str] = [],
        decrease_keys: List[str] = [],
        tolerance: float = 0.1,
    ):
        self._use_rollback = True
        self._rollback_monitor = _ModelMonitor(
            logger=self.logger, 
            increase_keys=increase_keys,
            decrease_keys=decrease_keys,
            tolerance=tolerance,
        )
        self.logger.info(f"Rollback Monitor enabled (tol={tolerance})")

    def monitor_check(self, step: int, status: dict):

        if not getattr(self, '_use_rollback', False):
            return False, None, -1

        degraded = self._rollback_monitor(step=step, status=status)
        _best_cp_path = self.fpath / f"checkpoint_{self._rollback_monitor.best_step}"
        return degraded, _best_cp_path, self._rollback_monitor.best_step

    def add_stats(self, commit: bool = True, **stats):
        _key_accurate_stats = self._flatten_dict(stats)

        if commit:
            if 'step' in _key_accurate_stats:
                self._step = _key_accurate_stats['step']
            else:
                # Auto-tick the clock
                self._step += 1

        for k, v in _key_accurate_stats.items():
            if self._use_wandb and k not in self._wandb_defined_metrics:
                wandb.define_metric(name=k, step_metric="step")
                self._wandb_defined_metrics.add(k)
            self._stats.setdefault(k, []).append(v)

        if "step" not in _key_accurate_stats:
            _key_accurate_stats["step"] = self._step

        if self._use_wandb:
            wandb.log(_key_accurate_stats, step=self._step)

    def add_config(self, **config):
        clean_config = self._sanitize_config(config)
        self._config.update(clean_config)
        self.save_config(self.fpath)

    def add_analysis(self, name: str, data: Dict[str, List], step: Optional[int] = None):

        current_step = step if step is not None else self._step
        analysis_dir = self._last_used_path
        if analysis_dir is None:
            analysis_dir = self.fpath / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

        csv_path = analysis_dir / f"{name}_analysis.csv"
        keys = list(data.keys())
        values = list(data.values())

        if not values: return
        rows = list(zip(*values))

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(rows)

        if self._use_wandb:
            table = wandb.Table(columns=keys, data=rows)
            wandb.log({f"analysis/{name}_{current_step}": table}, step=current_step)

    def add_image(self, name: str, image: Any, step: Optional[int] = None, **kwargs):

        current_step = step if step is not None else self._step

        if self._output_path is not None:
            save_dir = Path(self._output_path)
        else:
            save_dir = self.fpath / "outputs"
            save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / name

        if hasattr(image, 'savefig'):
            image.savefig(file_path, **kwargs)
        elif hasattr(image, 'save'):
            # PIL style
            image.save(file_path, **kwargs)
        else:
            self.logger.error("Object passed to add_image has no .save() or .savefig() method.")
            return

        if self._use_wandb:
            w_img = wandb.Image(str(file_path), caption=f"{name} (Step {current_step})")
            wandb.log({f"images/{name}": w_img}, step=current_step)

    def add_video(self, name: str, video_path: str, step: Optional[int] = None):

        current_step = step if step is not None else self._step

        if self._output_path:
            save_dir = Path(self._output_path)
        else:
            save_dir = self.fpath / "outputs"
            save_dir.mkdir(parents=True, exist_ok=True)

        src = Path(video_path)
        if not src.exists():
            self.logger.error(f"Video file {src} not found.")
            return

        extension = src.suffix # e.g. .mp4 or .gif
        filename = f"{name}_{current_step}{extension}"
        dst = save_dir / filename

        if self._use_wandb:
            fmt = extension.lstrip(".")
            w_vid = wandb.Video(str(src), caption=f"{name} (Step {current_step})", format=fmt)
            wandb.log({f"videos/{name}": w_vid}, step=current_step)

    def save_config(self, fpath: str) -> None:
        with open(f'{fpath}/config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper)
        if self._use_wandb:
            wandb.config.update(self._config, allow_val_change=True)

    def save_stats(self, fpath: Optional[str] = None) -> None:
        with open(f'{fpath if fpath else self._last_path}/stats.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self._stats, f, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper)

    def load_stats(self) -> dict:
        with open(self.fpath / "stats.yaml", "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def last_path(self) -> str:
        return self._last_used_path

    @property
    def output_path(self) -> str:
        return self._output_path

    @property
    def stats(self) -> dict:
        return self._stats

    @staticmethod
    def _create_dict(obj: object, key: str) -> Dict[str, torch.Tensor]:
        if obj is None:
            return { key: None}
        if not isinstance(obj, (list, tuple)):
            return { key: obj.state_dict()}
        else:
            out_dict = {}
            for i, o in enumerate(obj, start=1):
                out_dict.update({f'{key}_{i}': o.state_dict()})
            return out_dict

    def _setup_logger(self):
        logger = logging.getLogger(self.fpath.name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # file_handler = logging.FileHandler(self.fpath / "system.log")
        # file_handler.setFormatter(fmt)
        # logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        return logger

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(ModelLogger._flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

    @staticmethod
    def _sanitize_config(data: Any) -> Any:
            if isinstance(data, dict):
                return {k: ModelLogger._sanitize_config(v) for k, v in data.items()}
            if isinstance(data, (list, tuple)):
                return [ModelLogger._sanitize_config(v) for v in data]
            if isinstance(data, (numbers.Number, np.ndarray, bool, type(None))):
                return data
            return str(data)
