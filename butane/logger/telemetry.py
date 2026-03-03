import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

class WandbManager:
    def __init__(self, fpath: Path, overwrite: bool, resume: bool, logger: logging.Logger):
        import wandb
        self.wandb = wandb
        self.fpath = fpath
        self.logger = logger
        self.defined_metrics = set()

        self._init_run(overwrite, resume)

    def _init_run(self, overwrite: bool, resume: bool):
        project = os.environ.get("WANDB_PROJECT")
        name = os.environ.get("WANDB_RUN")
        assert project is not None, "Set the WANDB_PROJECT env variable"

        id_file = self.fpath / "wandb_id.txt"
        lineage_ids = [line.strip() for line in id_file.read_text().split('\n') if line.strip()] if id_file.exists() else []

        if overwrite and lineage_ids:
            self._delete_cloud_lineage(project, lineage_ids)
            lineage_ids = []

        if resume and not overwrite and lineage_ids:
            old_id = lineage_ids[-1]
            try:
                old_name = self.wandb.Api().run(f"{project}/{old_id}").name
            except Exception:
                old_name = name if name else f"{self.fpath.name}_{old_id}"
            run_name = f"{old_name}_resume" 
        else:
            run_id = self.wandb.util.generate_id()
            run_name = f"{self.fpath.name}_{run_id}" if name is None else name

        mode = "a" if (resume and not overwrite) else "w"
        prefix = "\n" if (resume and not overwrite and lineage_ids) else ""
        with open(id_file, mode) as f:
            f.write(f"{prefix}{run_name}")

        self.wandb.init(
            project=project, name=run_name, id=run_name, dir=str(self.fpath),
            settings=self.wandb.Settings(_disable_stats=True, _disable_meta=True)
        )
        self.wandb.define_metric("step", hidden=True)

    def _delete_cloud_lineage(self, project: str, lineage_ids: List[str]):
        self.logger.info(f"🗑️ Deleting {len(lineage_ids)} old WandB run(s)...")
        api = self.wandb.Api()
        for old_id in lineage_ids:
            try:
                api.run(f"{project}/{old_id}").delete()
            except Exception as e:
                self.logger.warning(f"Could not delete {old_id}: {e}")

    def define_metrics(self, flat_stats: dict):
        for k in flat_stats.keys():
            if k not in self.defined_metrics and k != "step":
                self.wandb.define_metric(k, step_metric="step")
                self.defined_metrics.add(k)

    def log_metrics(self, entry: dict, step: int):
        self.wandb.log(entry, step=step, commit=True)

    def update_config(self, config: dict):
        self.wandb.config.update(config, allow_val_change=True)

    def replay_history(self, current_step: int, history_columns: dict):
        self.logger.info("Replaying historical stats into the new WandB branch...")

        # Invert column-based stats into row-based payloads for replay
        replay_timeline = {}
        for k, v_list in history_columns.items():
            if k == "step" or k.endswith("/step"): continue
            for i, val in enumerate(v_list):
                if val is None: continue
                s = history_columns.get('step', [])[i] if 'step' in history_columns else i + 1
                if s <= current_step:
                    replay_timeline.setdefault(s, {})[k] = val

        sorted_steps = sorted(replay_timeline.keys())
        for i, s in enumerate(sorted_steps):
            self.wandb.log(replay_timeline[s], step=s)
            if i > 0 and i % 5000 == 0: time.sleep(0.5) # Prevent CommError

        self.logger.info(f"✅ Replayed {len(sorted_steps)} steps of history into WandB.")

    # Artifact specific logs
    def log_plot(self, name: str, plot: Any, step: int): self.wandb.log({f"plot/{name}": plot}, step=step)
    def log_image(self, name: str, path: Path, step: int): self.wandb.log({f"images/{name}": self.wandb.Image(str(path))}, step=step)
    def log_video(self, name: str, path: Path, step: int): self.wandb.log({f"videos/{name}": self.wandb.Video(str(path), format="mp4")}, step=step)
    def log_table(self, name: str, data: dict, step: int):
        keys, rows = list(data.keys()), list(zip(*list(data.values())))
        self.wandb.log({f"analysis/{name}_{step}": self.wandb.Table(columns=keys, data=rows)}, step=step)
