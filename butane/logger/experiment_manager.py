import json
import logging
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

class ExperimentEnvironment:
    """ Helper class that manages the local experiment directories and paths """
    def __init__(self, fpath: str, overwrite: bool, resume: bool, eval_mode: bool, logger: logging.Logger):
        self.fpath = Path(fpath)
        self.overwrite = overwrite
        self.resume = resume
        self.eval_mode = eval_mode
        self.logger = logger

        self.last_path: Optional[Path] = None
        self.output_path: Optional[Path] = None

        self._setup_directory()

    def _setup_directory(self) -> None:

        if self.overwrite and self.fpath.exists():
            print(f"\n⚠️ WARNING: Overwrite is set to True for '{self.fpath}'.")
            print("This will completely DESTROY existing local stats and delete the WandB lineage from the cloud.")
            ans = input("Proceed? [y/N]: ")
            if ans.strip().lower() != 'y':
                print("Aborted by user.")
                sys.exit(0)

        if self.fpath.exists():
            if not (self.resume or self.eval_mode or self.overwrite):
                mtime = self.fpath.stat().st_mtime
                ts = datetime.datetime.fromtimestamp(mtime).strftime('%Y_%m_%d__%H_%M_%S')
                archived_path = self.fpath.with_name(f"{self.fpath.name}_{ts}")
                self.fpath.rename(archived_path)
                self.logger.info(f"Archived previous run to: {archived_path}")
        self.fpath.mkdir(parents=True, exist_ok=True)

    def update_paths(self, checkpoint_path: Path, output_path: Path) -> None:
        self.last_path = checkpoint_path
        self.output_path = output_path


class HistoryManager:

    def __init__(self, fpath: Path, logger: logging.Logger):
        self.fpath = fpath
        self.logger = logger
        self.stats_file = self.fpath / "stats.jsonl"
        self._stage_buffer: Dict[str, Any] = {}
        self.stats: Dict[str, List] = {} # Populated only during recovery for WandB replay

    def stage_metrics(self, stats: dict) -> Dict[str, Any]:
        flat_stats = self._flatten_dict(stats)
        self._stage_buffer.update(flat_stats)
        return flat_stats

    def flush_buffer_to_jsonl(self, step: int) -> Dict[str, Any]:
        entry = {"step": step}
        for k, v in self._stage_buffer.items():
            entry[k] = v.item() if hasattr(v, 'item') else v

        with open(self.stats_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self._stage_buffer.clear()
        return entry

    def recover_state(self, safe_step: int, overwrite: bool) -> None:
        if overwrite:
            self.logger.info("Overwrite is True: Wiping previous stats history.")
            if self.stats_file.exists():
                self.stats_file.unlink()
            self.stats = {}
            return

        # Load available history into RAM for WandB replay later
        self.stats = self._load_jsonl_to_columns()

        # Truncate orphaned rows on disk
        if self.stats_file.exists():
            valid_lines = []
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    if row.get("step", 0) <= safe_step:
                        valid_lines.append(line)

            temp_file = self.stats_file.with_suffix('.jsonl.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.writelines(valid_lines)
            temp_file.replace(self.stats_file)

        self.logger.info(f"History safely truncated to step {safe_step} on disk.")

    def reset(self):
        self.stats = {}

    def _load_jsonl_to_columns(self) -> dict:
        if not self.stats_file.exists(): return {}
        columns = {}
        with open(self.stats_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                for k, v in row.items():
                    columns.setdefault(k, []).append(v)
        return columns

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(HistoryManager._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
