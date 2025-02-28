import copy
import torch

class Ensemble(torch.nn.Module):

    def __init__(self, model: torch.nn.Module, depth: int) -> None:
        super().__init__()
        self.__base_model = model
        self._ensemble_depth = depth
        self.ensemble = [copy.deepcopy(self.__base_model) for _ in range(self._ensemble_depth)]

    def optimizer(self, optim, **kwargs):
        self.optimizers = [optim(model.parameters(), **kwargs) for model in self.ensemble]

    def eval(self):
        for model in self.ensemble:
            model.eval()
        self.training = False

    def train(self):
        for model in self.ensemble:
            model.train()
        self.training = True

    def __iter__(self):
        if self.training:
            for model, optimizer in zip(self.ensemble, self.optimizers):
                yield model, optimizer
        else:
            for model in self.ensemble:
                yield model

    @staticmethod
    def __reset_weights(model):
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
