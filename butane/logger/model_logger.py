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

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False


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
    """
    Initializes the ModelLogger for experiment tracking, checkpointing, and cloud synchronization.

    The logger manages a state machine via 'resume', 'overwrite', and 'eval_mode' to ensure
    local and remote (WandB) data consistency.

    Args:
        fpath (str): The root directory for the experiment.
        overwrite (bool): If True, wipes local stats and deletes the entire WandB lineage 
            from the server. Prompts for user verification if the folder exists.
        resume (bool): If True, attempts to load existing stats and continue the experiment. 
            Creates a new 'branch' in WandB to avoid data overlap.
        eval_mode (bool): If True, locks onto the existing folder without creating 
            timestamps and disables all WandB logging. Perfect for inference scripts.
        use_wandb (bool): Enables Weights & Biases integration. Requires 'WANDB_PROJECT' 
            env variable.

    Internal State Matrix:
        - Safe Start (False, False, False): Creates a timestamped folder and a new WandB run.
        - Recovery (False, True, False): Re-attaches to 'fpath', loads weights/stats, 
            truncates 'stats.yaml' to remove post-checkpoint data, branches the WandB 
            run with a lineage-tracking name, and replays history into the new run.
        - Hard Reset (True, True, False): Loads weights from 'fpath', but deletes ALL 
            previous WandB runs in the lineage and wipes local 'stats.yaml' history.
        - Evaluation (False, False, True): Static access to 'fpath' for weight loading; 
            no logging or directory mutation.

    WandB Lineage Tracking:
        The logger maintains a 'wandb_id.txt' file in 'fpath'. This acts as a 'hit-list'
        for overwrites and a 'breadcrumb' for resumes. Resumed runs are renamed on the
        server to "ParentName_resume_from_step_X" to maintain clear experiment history.
    """

    def __init__(
        self,
        fpath: str,
        overwrite: bool = False,
        resume: bool = False,
        eval_mode: bool = False,
        use_wandb: bool = False,
    ):
        self.fpath = Path(fpath)
        self._overwrite = overwrite
        self._resume = resume
        self._eval_mode = eval_mode

        if self._overwrite and self.fpath.exists():
            print(f"\n⚠️ WARNING: Overwrite is set to True for '{self.fpath}'.")
            print("This will completely DESTROY existing local stats and delete the WandB lineage from the cloud.")
            ans = input("Proceed? [y/N]: ")
            if ans.strip().lower() != 'y':
                print("Aborted by user.")
                sys.exit(0)

        if self.fpath.exists() and not self._overwrite and not self._resume and not self._eval_mode:
            ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            self.fpath = self.fpath.with_name(f"{self.fpath.name}_{ts}")
        self.fpath.mkdir(parents=True, exist_ok=True)

        self._use_rollback = False
        self._rollback_monitor = None

        self.logger = self._setup_logger()
        self.logger.info(f"Initialized Experiment at: {self.fpath}")

        self._stats = {}
        self._config = {}
        self._stage_buffer = {}
        self._last_used_path, self._output_path = None, None
        self._step = 1

        self._use_wandb = _HAS_WANDB and use_wandb and not self._eval_mode
        self._old_wandb_id = None

        if self._use_wandb:
            self._init_wandb()

    def checkpoint(
        self,
        *,
        model: Optional[ModuleParams] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        ema: ModuleParams = None,
        **modules,
    ):

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

        _path = self.fpath / f"checkpoint_{self._step}"
        output_path = _path / "outputs/"
        output_path.mkdir(parents=True, exist_ok=True)

        cp = dict(
            step=self._step,
            **self._create_dict(model, "model"),
            **self._create_dict(optimizer, "optimizer"),
            **self._create_dict(lr_scheduler, "lr_scheduler"),
            **self._create_dict(ema, "ema"),
        )
        for k,v in modules.items():
            cp.update(self._create_dict(v, k))

        if getattr(self, '_use_rollback', False) and self._rollback_monitor:
            cp['monitor_state'] = self._rollback_monitor.state()

        torch.save(cp, _path / "checkpoint.pt")

        self._last_used_path = _path
        self._output_path = str(output_path)
        self.logger.info(f"Checkpoint saved: checkpoint_{self._step}")

    def load_checkpoint(
        self,
        step: int,
        *,
        model: ModuleParams,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        ema: ModuleParams = None,
        **modules,
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
            ema=ema,
            **modules,
        )

        loaded_step = checkpoint.get("step", step)

        if self._resume:
            self._step = loaded_step + 1

            if self._overwrite:
                self.logger.info("Overwrite is True: Wiping previous stats history.")
                self._stats = {'step': loaded_step}
            else:
                try:
                    self._stats = self.load_stats()
                except FileNotFoundError:
                    self._stats = {}

                truncated_stats = {}
                for k, v in self._stats.items():
                    if k == "step" or k.endswith("/step"): continue

                    if '/' in k:
                        group_step_key = f"{k.split('/')[0]}/step"
                        if group_step_key not in self._stats: continue
                        group_step = self._stats[group_step_key]

                        if not group_step: continue
                        if loaded_step < group_step[0]: continue

                        valid_steps = [x for x in group_step if x <= loaded_step]
                        if not len(valid_steps): continue

                        closest_step = max(valid_steps)
                        max_idx_to_keep = group_step.index(closest_step) + 1
                        truncated_stats[k] = v[:max_idx_to_keep]
                        truncated_stats[group_step_key] = group_step[:max_idx_to_keep]
                    else:
                        t_list = v[:loaded_step]
                        while len(t_list) < loaded_step:
                            t_list.append(None)
                        truncated_stats[k] = t_list

                self._stats = truncated_stats
                self._stats['step'] = loaded_step

        else:
            self._step = loaded_step
            self._stats = {}

        if getattr(self, '_use_rollback', False) and self._rollback_monitor:
            monitor_state = checkpoint.get('monitor_state')
            if monitor_state:
                self._rollback_monitor.load_state_dict(monitor_state)
                self.logger.info(f"Rollback Monitor state restored (Best Step: {self._rollback_monitor.best_step})")

        if self._resume and not self._overwrite and getattr(self, '_use_wandb', False) and getattr(self, '_old_wandb_id', None):
            base_name = getattr(self, '_old_wandb_name', f"{self.fpath.name}_{self._old_wandb_id}")
            new_name = f"{base_name}_resume_from_step_{step}"

            wandb.run.name = new_name
            try:
                api = wandb.Api()
                project_name = wandb.run.project
                server_run = api.run(f"{project_name}/{wandb.run.id}")
                server_run.name = new_name
                server_run.update()
                self.logger.info(f"✨ WandB run renamed on server to: {new_name}")
            except Exception as e:
                self.logger.warning(f"Could not push name change to WandB server: {e}")

        # --- WANDB STATS REPLAY (Only if safe resuming) ---
        if self._resume and not self._overwrite and getattr(self, '_use_wandb', False) and self._stats:
            self.logger.info("Replaying historical stats into the new WandB branch...")
            replay_timeline = {}

            for k, v in self._stats.items():
                if k == "step" or k.endswith("/step"): 
                    continue

                if '/' in k:
                    group_step_key = f"{k.split('/')[0]}/step"
                    steps = self._stats.get(group_step_key, [])
                    for i, val in enumerate(v):
                        if val is None: continue
                        if i < len(steps):
                            s = steps[i]
                            if s not in replay_timeline: replay_timeline[s] = {}
                            replay_timeline[s][k] = val
                else:
                    for i, val in enumerate(v):
                        if val is None: continue
                        s = i + 1
                        if s not in replay_timeline: replay_timeline[s] = {}
                        replay_timeline[s][k] = val

            sorted_steps = sorted(replay_timeline.keys())
            for s in sorted_steps:
                payload = replay_timeline[s]
                payload["step"] = s
                wandb.log(payload)
            self.logger.info(f"✅ Replayed {len(sorted_steps)} steps of history into WandB.")

        if ckpt_folder.exists():
            self._last_used_path = ckpt_folder
            self._output_path = str(ckpt_folder / "outputs")
            self.logger.info(f"State restored from step {loaded_step}. Internal clock: {self._step}.")
        return checkpoint

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

    def commit(self, step: Optional[int] = None):

        commit_step = step if step is not None else self._step
        self._stats["step"] = commit_step

        updated_groups = set()
        for k, v in self._stage_buffer.items():
            val = v.item() if hasattr(v, 'item') else v
            self._stats.setdefault(k, []).append(val)

            # ungrouped metrics
            if '/' in k:
                group_prefix = k.split('/')[0]
                group_step_key = f"{group_prefix}/step"

                if group_step_key not in updated_groups:
                    self._stats.setdefault(group_step_key, []).append(self._step)
                    updated_groups.add(group_step_key)

        if self._use_wandb:
            payload = {**self._stage_buffer, "step": self._step}
            wandb.log(payload, commit=True)

        self.save_stats(self.fpath)
        self._stage_buffer.clear()
        if step is not None:
            self._step = step + 1
        else:
            self._step += 1

    def add_stats(self, **stats):
        flat_stats = self._flatten_dict(stats)
        self._stage_buffer.update(flat_stats)
        if self._use_wandb:
            for k in flat_stats.keys():
                if k not in self._wandb_defined_metrics:
                    wandb.define_metric(k, step_metric="step")
                    self._wandb_defined_metrics.add(k)

    def add_config(self, **config):
        clean_config = self._sanitize_config(config)
        self._config.update(clean_config)
        self.save_config(self.fpath)

    def add_analysis(self, name: str, data: Dict[str, List]):

        current_step = self._step
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

    def add_plot(self, name: str, plot: Any, **kwargs):

        current_step = self._step
        if self._output_path is not None:
            save_dir = Path(self._output_path)
        else:
            save_dir = self.fpath / "outputs"
            save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / name

        if hasattr(plot, 'savefig'):
            plot.savefig(file_path, **kwargs)
        else:
            self.logger.error("Object passed to add_plot has no .save() or .savefig() method.")
            return

        if self._use_wandb:
            wandb.log({f"plot/{name}": plot}, step=current_step)

    def add_image(self, name: str, image: Any, **kwargs):

        current_step = self._step
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

    def add_video(
        self,
        name: str,
        video: Union[str, Path, np.ndarray],
    ) -> None:

        current_step = self._step
        if self._output_path:
            save_dir = Path(self._output_path)
        else:
            save_dir = self.fpath / "outputs"
            save_dir.mkdir(parents=True, exist_ok=True)

        fmt = "mp4"
        if isinstance(video, (str, Path)):
            source_path = Path(video)
            if not source_path.exists():
                self.logger.error(f"Video file {source_path} not found.")
                return

            extension = source_path.suffix
            fmt = extension.lstrip(".")
            target_filename = f"{name}_{current_step}{extension}"
            target_path = save_dir / target_filename

            if source_path.absolute() != target_path.absolute():
                shutil.copy(source_path, target_path)

            local_video_path = target_path

        else:
            if not _HAS_IMAGEIO:
                self.logger.warning("ImageIO lib does not exist; Video will not be created.")
                return

            target_filename = f"{name}_{current_step}.mp4"
            target_path = save_dir / target_filename

            imageio.mimwrite(target_path, video, fps=30, quality=8)
            local_video_path = target_path

        if self._use_wandb:
            w_vid = wandb.Video(
                str(local_video_path),
                caption=f"{name} (Step {current_step})",
                fps=30,
                format=fmt
            )
            wandb.log({f"videos/{name}": w_vid}, step=self._step)

    def set_step(self, step: int):
        self._step = step

    def step(self):
        self._step += 1
        if self._use_wandb:
            wandb.log({"step": self._step}, step=self._step)

    def save_config(self, fpath: str) -> None:
        with open(f'{fpath}/config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper)
        if self._use_wandb:
            wandb.config.update(self._config, allow_val_change=True)

    def save_stats(self, fpath: Optional[str] = None) -> None:
        with open(f'{fpath if fpath else self._last_used_path}/stats.yaml', 'w', encoding='utf-8') as f:
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
            assert hasattr(obj, "state_dict"), "Object does not have state_dict()"
            return { key: obj.state_dict()}
        else:
            out_dict = {}
            for i, o in enumerate(obj, start=1):
                assert hasattr(o, "state_dict"), "Object does not have state_dict()"
                out_dict.update({f'{key}_{i}': o.state_dict()})
            return out_dict

    def _setup_logger(self):
        logger = logging.getLogger(self.fpath.name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        return logger

    def _init_wandb(self):
        project = os.environ.get("WANDB_PROJECT")
        name = os.environ.get("WANDB_RUN")
        assert project is not None, "Set the WANDB_PROJECT env variable"

        id_file = self.fpath / "wandb_id.txt"
        self._wandb_defined_metrics = set()

        lineage_ids = []
        if id_file.exists():
            lineage_ids = [line.strip() for line in id_file.read_text().strip().split('\n') if line.strip()]

        # 1. OVERWRITE: ALWAYS delete all runs in the lineage (even if resuming weights)
        if self._overwrite and lineage_ids:
            self.logger.info(f"🗑️ Overwrite=True: Deleting {len(lineage_ids)} old WandB run(s) from server...")
            try:
                api = wandb.Api()
                for old_id in lineage_ids:
                    try:
                        run = api.run(f"{project}/{old_id}") 
                        run.delete()
                        self.logger.info(f"   -> Deleted run {old_id}")
                    except Exception as e:
                        self.logger.warning(f"   -> Could not delete {old_id}: {e}")
            except Exception as e:
                self.logger.warning(f"WandB API Error during deletion: {e}")
            lineage_ids = [] # Clear lineage locally

        # 2. RESUME (Only if NOT overwriting): Setup branching
        if self._resume and not self._overwrite and lineage_ids:
            self._old_wandb_id = lineage_ids[-1] 
            try:
                api = wandb.Api()
                old_run = api.run(f"{project}/{self._old_wandb_id}")
                self._old_wandb_name = old_run.name
            except Exception:
                self._old_wandb_name = name if name else f"{self.fpath.name}_{self._old_wandb_id}"

        # 3. GENERATE & SAVE: Create the new run ID
        run_id = wandb.util.generate_id()
        run_name = f"{self.fpath.name}_{run_id}" if name is None else name

        mode = "a" if (self._resume and not self._overwrite) else "w"
        prefix = "\n" if (self._resume and not self._overwrite and lineage_ids) else ""
        with open(id_file, mode) as f:
            f.write(f"{prefix}{run_id}")

        self.logger.info(f"Staged NEW WandB Run ID: {run_id}")

        wandb.init(
            project=project,
            name=run_name,
            id=run_id,
            dir=str(self.fpath),
        )

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
