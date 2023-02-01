from typing import Any
from abc import ABC, abstractmethod

import torch
from torch import nn


class Model(ABC):
    def __call__(self, input: Any):
        return self._inference(input)

    @abstractmethod
    def _inference(self, input: Any):
        pass


class TorchModel(Model):
    def __init__(self, architecture: nn.Module):
        self._model = model

    def __call__(self, input: torch.Tensor):
        return self._inference(input=input)

    def _inference(self, input: torch.Tensor):
        return self._model(input)

    def on_train_epoch_start(self):
        self._model.train()

    def on_val_epoch_start(self):
        self._model.eval()
