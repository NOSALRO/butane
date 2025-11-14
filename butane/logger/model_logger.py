from typing import Optional, List, Dict
from pathlib import Path
import numbers
import json
import datetime
import os
import re
import torch
import numpy as np
import yaml
from yaml.representer import SafeRepresenter
from .._typedefs import *

from .. import nn


class _LiteralDumper(yaml.SafeDumper):
    pass

def _multiline_str_presenter(dumper: yaml.SafeDumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)
_LiteralDumper.add_representer(str, _multiline_str_presenter)

class _ModelMonitor:

    def __init__(
        self,
        *,
        increase_keys: List[str] = [],
        decrease_keys: List[str] = [],
        tolerance: float = 0.1,
    ):

        self._increase_keys = increase_keys
        self._decrease_keys = decrease_keys
        self._tolerance = tolerance
        self.best_epoch = -1

    def __call__(self, epoch: int, status: dict): 

        if self.best_epoch < 0:
            self.best_epoch = epoch
            self.best_metrics = status
            return False

        degraded = self._check_degradation(status)
        if degraded:
            print(f"[Monitor] Epoch {epoch}: Metrics degraded > {self._tolerance*100:.0f}%")
            return True
        else:
            self.best_epoch = epoch
            self.best_metrics = status
            print(f"[Monitor] Epoch {epoch}: Metrics OK")
            return False

    def _check_degradation(self, current: Dict[str, float]) -> bool:

        for ik in self._increase_keys:
            best, new = self.best_metrics[ik], current[ik]
            if new < best * (1 - self._tolerance):
                return True

        for dk in self._decrease_keys:
            best, new = self.best_metrics[dk], current[dk]
            if new > best * (1 + self._tolerance):
                return True

        return True

class ModelLogger:

    def __init__(
        self,
        fpath: str,
        overwrite: bool = False,
    ):
        self.fpath = Path(fpath)
        self.__overwrite = overwrite

        if self.fpath.exists() and not self.__overwrite:
            ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            self.fpath = self.fpath.with_name(f"{self.fpath.name}_{ts}")
        self.fpath.mkdir(parents=True, exist_ok=True)


        self.log = {}
        self.__stats = {}
        self.__last_used_path, self.__output_path = None, None

    def enable_rollback(
        self,
        increase_keys: List[str] = [],
        decrease_keys: List[str] = [],
        tolerance: float = 0.1,
    ):
        self._use_rollback = True
        self._rollback_monitor = _ModelMonitor(
            increase_keys=increase_keys,
            decrease_keys=decrease_keys,
            tolerance=tolerance,
        )

    def monitor_check(
        self,
        epoch: int,
        status: dict,
    ):
        if self._use_rollback:
            _flag = self._rollback_monitor(epoch=epoch, status=status)
            _best_cp_path = f"{self.fpath}/checkpoint_{self._rollback_monitor.best_epoch}"
            return _flag, _best_cp_path, self._rollback_monitor.best_epoch

    def checkpoint(
        self,
        epoch: int,
        *,
        model: ModuleParams,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        ema: ModuleParams = None,
        scaler: Optional[torch.nn.Module] = None,
    ):

        assert isinstance(epoch, int), f"'epoch' must be int, got {type(epoch).__name__}"
        is_mod = lambda o: isinstance(o, torch.nn.Module)
        is_mod_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.nn.Module) for x in o)
        is_opt = lambda o: isinstance(o, torch.optim.Optimizer)
        is_opt_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.optim.Optimizer) for x in o)
        is_sched = lambda o: isinstance(o, torch.optim.lr_scheduler.LRScheduler)
        is_sched_list = lambda o: isinstance(o, (list, tuple)) and all(isinstance(x, torch.optim.lr_scheduler.LRScheduler) for x in o)

        assert is_mod(model) or is_mod_list(model), "`model` must be a torch.nn.Module or a list/tuple of torch.nn.Modules."

        if optimizer is not None:
            assert is_opt(optimizer) or is_opt_list(optimizer), "`optimizer` must be a torch.optim.Optimizer or list/tuple of them."
        if lr_scheduler is not None:
            assert is_sched(lr_scheduler) or is_sched_list(lr_scheduler), "`lr_scheduler` must be a torch.optim.lr_scheduler.LRScheduler or list/tuple of them."
        if ema is not None:
            assert is_mod(ema) or is_mod_list(ema), "`ema` must be a torch.nn.Module or list/tuple of torch.nn.Modules."
        if scaler is not None:
            assert isinstance(scaler, torch.nn.Module), "`scaler` must be a torch.nn.Module or None."

        _path = f"{self.fpath}/checkpoint_{epoch}/"
        output_path = Path(_path, "outputs/")
        output_path.mkdir(parents=True, exist_ok=True)

        cp = dict(
            epoch=epoch,
            **self.__create_dict(model, "model"),
            **self.__create_dict(optimizer, "optimizer"),
            **self.__create_dict(lr_scheduler, "lr_scheduler"),
            **self.__create_dict(ema, "ema"),
            **self.__create_dict(scaler, "scaler"),
        )
        torch.save(cp, _path + "checkpoint.pt")

        self.save()

        self.__last_used_path = _path
        self.__output_path = str(output_path)

    def load_checkpoint(
        self,
        epoch: int,
        *,
        model: ModuleParams,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        ema: ModuleParams = None,
        scaler: Optional[torch.nn.Module] = None,
    ):
        nn.utils.load_state(
            str(self.fpath.absolute()) + f"/checkpoint_{epoch}",
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema, scaler=scaler
        )

    def add_stats(self, **stats):
        for k, v in stats.items():
            self.__stats.setdefault(k, []).append(v)

    def add_logs(self, **logs):
        for k, v in logs.items():
            if not k in self.log:
                if not isinstance(v, (str, dict, list, numbers.Number)):
                    v = str(v)
                self.log[k] = v

    def save(self):
        self.save_log(self.fpath)
        self.save_stats(self.fpath)


    def save_log(self, fpath: Optional[str] = None) -> None:
        with open(f'{fpath if fpath else self.last_path}/log.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.log, f, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper)

    def save_stats(self, fpath: Optional[str] = None) -> None:
        with open(f'{fpath if fpath else self.last_path}/stats.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.__stats, f, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper)

    def load_stats(self) -> dict:
        with open(self.fpath / "stats.yaml", "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def __create_dict(self, obj: object, key: str) -> Dict[str, torch.Tensor]:
        if obj is None:
            return { key: None}
        if not isinstance(obj, (list, tuple)):
            return { key: obj.state_dict()}
        else:
            out_dict = {}
            for i, o in enumerate(obj, start=1):
                out_dict.update({f'{key}_{i}': o.state_dict()})
            return out_dict

    @property
    def last_path(self) -> str:
        return self.__last_used_path

    @property
    def output_path(self) -> str:
        return self.__output_path

    @property
    def stats(self) -> dict:
        return self.__stats
