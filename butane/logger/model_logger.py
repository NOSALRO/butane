from typing import Optional, List
import json
import datetime
import os
import re
import torch
from .._typedefs import *


class ModelLogger:

    def __init__(self, fpath: str, overwrite: Optional[bool] = False):
        if fpath[-1] == "/":
            fpath = fpath[:-1]
        self.fpath = fpath
        self.log = {}
        self.__last_used_path = None
        self.__created = False
        self.__overwrite = overwrite

    def __init_log(self, fpath: str):
        if not os.path.exists(self.fpath):
            os.mkdir(self.fpath)
        else:
            if not self.__overwrite:
                self.fpath = f"{fpath}_{datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}/"
                os.mkdir(self.fpath)
        self.__created = True

    def checkpoint(self, id: int, model: ModuleParams, optimizer: Optional[torch.optim.Optimizer] = None):

        if not (isinstance(model, torch.nn.Module) or isinstance(model, (list, tuple))):
            raise ("Model should be either torch.nn.Module or list of torch.nn.Modules")

        if not self.__created:
            self.__init_log(self.fpath)

        _path = f"{self.fpath}/checkpoint_{id}/"
        if not os.path.exists(_path):
            os.mkdir(_path)
            os.mkdir(_path + "/outputs")

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

        if optimizer is not None:
            torch.save(optimizer.state_dict(), f"{_path}/optimizer.pt")
            self.log['optimizer']= {'lr': optimizer.param_groups[0]['lr']}

        with open(f'{_path}/log.json', 'w', encoding='utf-8') as f:
            json.dump(self.log, f, ensure_ascii=False, indent=4)
        self.__last_used_path = _path

    @property
    def last_path(self):
        return self.__last_used_path

    @property
    def output_path(self):
        return self.__last_used_path + "/outputs"
