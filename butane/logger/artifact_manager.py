import csv
import yaml
import numbers
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

class _LiteralDumper(yaml.SafeDumper): pass
def _multiline_str_presenter(dumper: yaml.SafeDumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)
_LiteralDumper.add_representer(str, _multiline_str_presenter)

class ArtifactManager:
    def __init__(self, fpath: Path, logger: logging.Logger):
        self.fpath = fpath
        self.logger = logger
        self._config = {}

    def save_config(self, config: dict) -> None:
        clean_config = self._sanitize_config(config)
        self._config.update(clean_config)
        with open(self.fpath / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper)

    def save_csv(self, name: str, data: Dict[str, List], analysis_dir: Optional[Path]) -> Optional[Path]:
        target_dir = analysis_dir if analysis_dir else (self.fpath / "analysis")
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / f"{name}_analysis.csv"

        keys, values = list(data.keys()), list(data.values())
        if not values: return None

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(list(zip(*values)))
        return csv_path

    def save_media(self, name: str, obj: Any, out_dir: Optional[Path], **kwargs) -> Optional[Path]:
        """ Mainly for plots and images thourgh matplotlib or seaborn """
        target_dir = out_dir if out_dir else (self.fpath / "outputs")
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / name

        if hasattr(obj, 'savefig'): obj.savefig(file_path, **kwargs)
        elif hasattr(obj, 'save'): obj.save(file_path, **kwargs)
        else:
            self.logger.error(f"Object {name} has no .save() or .savefig() method.")
            return None
        return file_path

    def save_video(self, name: str, video: Union[str, Path, np.ndarray], out_dir: Optional[Path], **imageio_kwargs) -> Optional[Path]:
        target_dir = out_dir if out_dir else (self.fpath / "outputs")
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / name
        if target_path.suffix != '.mp4': 
            target_path = target_path.with_suffix('.mp4')

        if isinstance(video, (str, Path)):
            source_path = Path(video)
            if not source_path.exists():
                self.logger.error(f"Video file {source_path} not found.")
                return None
            if source_path.absolute() != target_path.absolute():
                shutil.copy(source_path, target_path)
        else:
            import imageio
            if 'fps' not in imageio_kwargs: imageio_kwargs["fps"] = 30
            if 'quaility' not in imageio_kwargs: imageio_kwargs["quaility"] = 8
            imageio.mimwrite(target_path, video, **imageio_kwargs)

        return target_path

    @staticmethod
    def _sanitize_config(data: Any) -> Any:
        if isinstance(data, dict): return {k: ArtifactManager._sanitize_config(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)): return [ArtifactManager._sanitize_config(v) for v in data]
        if isinstance(data, (numbers.Number, np.ndarray, bool, type(None))): return data
        return str(data)
