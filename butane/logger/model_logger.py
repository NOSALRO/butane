from typing import Optional, List
from pathlib import Path
import numbers
import json
import datetime
import os
import re
import torch
import yaml
from yaml.representer import SafeRepresenter
from .._typedefs import *



class _LiteralDumper(yaml.SafeDumper):
    pass

def _multiline_str_presenter(dumper: yaml.SafeDumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)
_LiteralDumper.add_representer(str, _multiline_str_presenter)


class ModelLogger:

    def __init__(self, fpath: str, overwrite: Optional[bool] = False):
        if fpath[-1] == "/":
            fpath = fpath[:-1]
        self.fpath = fpath
        self.log = {}
        self.__last_used_path = None
        self.__created = False
        self.__overwrite = overwrite
        self.__stats = {}

    def __init_log(self):
        p = Path(self.fpath)
        if p.exists() and not self.__overwrite:
            ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            p = p.with_name(f"{p.name}_{ts}")
        p.mkdir(parents=True, exist_ok=True)
        self.fpath = str(p)
        self.__created = True

    def add_stats(self, **stats):
        for k, v in stats.items():
            if not k in list(self.__stats.keys()):
                self.__stats[k] = []
            self.__stats[k].append(v)

    def add_logs(self, **logs):
        for k, v in logs.items():
            if not k in list(self.log.keys()):
                if not isinstance(v, (str, dict, list, numbers.Number)):
                    v = str(v)
                self.log[k] = v

    def checkpoint(
        self,
        checkpoint_id: int,
        *,
        model: ModuleParams,
        optimizer: Optional[torch.optim.Optimizer] = None,
        ema: ModuleParams = None,
        scaler: Optional[torch.nn.Module] = None,
    ):

        if not (isinstance(model, torch.nn.Module) or isinstance(model, (list, tuple))):
            raise ("Model should be either torch.nn.Module or list of torch.nn.Modules")

        if not self.__created:
            self.__init_log()

        _path = f"{self.fpath}/checkpoint_{checkpoint_id}/"
        p = Path(_path, "outputs/")
        p.mkdir(parents=True, exist_ok=True)

        # Save models
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), f"{_path}/model.pt")
            model_arch = open(f"{_path}/architecture.txt", "w")
            model_arch.write(f"{model}")
            model_arch.close()
        elif isinstance(model, (list, tuple)):
            for i, m in enumerate(model):
                torch.save(m.state_dict(), f"{_path}/model_{i}.pt")
                model_arch = open(f"{_path}/architecture_{i}.txt", "w")
                model_arch.write(f"{m}")
                model_arch.close()

        # Save EMA
        if ema is not None:
            if isinstance(ema, torch.nn.Module):
                torch.save(ema.state_dict(), f"{_path}/ema.pt")
            elif isinstance(model, (list, tuple)):
                for i, m in enumerate(ema):
                    torch.save(m.state_dict(), f"{_path}/ema_{i}.pt")

        if optimizer is not None:
            torch.save(optimizer.state_dict(), f"{_path}/optimizer.pt")
            self.log['optimizer']= {'lr': optimizer.param_groups[0]['lr']}

        if scaler is not None:
            torch.save(scaler.state_dict(), f"{_path}/scaler.pt")

        self.save_log(_path)
        self.save_stats(_path)

        self.__last_used_path = _path

    def save_log(self, fpath: Optional[str] = None) -> None:
        with open(f'{fpath if fpath else self.last_path}/log.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.log, f, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper)

    def save_stats(self, fpath: Optional[str] = None) -> None:
        with open(f'{fpath if fpath else self.last_path}/stats.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.__stats, f, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper)

    def load_stats(self, checkpoint_id: int) -> None:
        f = open(f"{self.fpath}/checkpoint_{checkpoint_id}/stats.yaml", 'r')
        data = yaml.load(f, Loader=yaml.SafeLoader)
        f.close()
        return data

    @property
    def last_path(self) -> str:
        return self.__last_used_path

    @property
    def output_path(self) -> str:
        return self.__last_used_path + "/outputs" if self.__last_used_path else None

    @property
    def stats(self) -> dict:
        return self.__stats
