from abc import ABC, abstractmethod

from core.model import TorchModel
import torch


class ModelSaver(ABC):
    def __call__(self, *args, **kwargs):
        self._save_model(*args, **kwargs)

    @abstractmethod
    def _save_model(self, *args, **kwargs):
        """This method saves the model."""
        pass


class TorchModelSaver(ModelSaver):
    def __init__(self):
        super().__init__()

    def _save_model(self, model: TorchModel):
        """This method saves a TorchModel"""

        torch.save(model.state_dict(), model.output_path)
