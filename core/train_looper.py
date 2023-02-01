import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import Dataset
from tqdm import tqdm

import config
from core.data_source import TorchDataSource
from core.model import TorchModel
from my_utils.evaluate import evaluate_dataset_patchwise


class TrainLooper(ABC):
    def __init__(self, **kwargs):

        self._history = {
            "train": {},
            "val": {},
        }

    def __call__(self, *args, **kwargs):
        return self._train_epoch(*args, **kwargs)

    @abstractmethod
    def train_epoch(self, *args, **kwargs):
        pass


class TorchLooper(TrainLooper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_batch(
        self, model: TorchModel, optimizer: Optimizer, criteria: _Loss, batch: Tuple
    ):

        input, labels = batch
        images = input.to(config.device)
        labels = labels.to(config.device)

        # Zero gradient before every batch
        optimizer.zero_grad()

        # Inference
        output = model(images)

        # Compute loss
        loss = criteria(output, labels)
        loss.backward()

        # Adjust weights
        optimizer.step()

        return {"loss": loss, "output": output, "labels": labels}

    def train_epoch(
        self,
        model: TorchModel,
        optimizer: Optimizer,
        criteria: _Loss,
        train_data_source: TorchDataSource,
    ):

        model.on_train_epoch_start()

        pbar = tqdm(train_data_source)
        for batch in pbar:

            result = self.train_batch(
                model=model, optimizer=optimizer, criteria=criteria, batch=batch
            )

            self._update_batch_history(result)

    @torch.no_grad()
    def val_batch(self, model: TorchModel, criteria: _Loss, batch: Tuple):

        input, labels = batch
        images = input.to(config.device)
        labels = labels.to(config.device)

        # Inference
        output = model(images)

        # Compute loss
        loss = criteria(output, labels)

        result = {"loss": loss, "output": output, "labels": labels}

        return result

    @torch.no_grad()
    def val_epoch(
        self,
        model: TorchModel,
        criteria: _Loss,
        val_data_source: TorchDataSource,
    ):

        model.on_val_epoch_start()

        pbar = tqdm(val_data_source)
        for batch in pbar:

            result = self.val_batch(model=model, criteria=criteria, batch=batch)

            self._update_batch_history(result, split="val")

    def _update_batch_history(self, result: Dict[Any, Any], split: str = "train"):

        if "batches_loss" in self._history.get(split, []):
            self._history[split]["batches_loss"] = {}

        i_batch = len(self._history[split]["batches_loss"])
        if self._history[split]["batches_loss"].get(i_batch) is None:
            self._history[split]["batches_loss"][i_batch] = []

        self._history[split]["batches_loss"][i_batch].append(result["loss"].item())


class ClBinarizationTorchLooper(TorchLooper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from torchmetrics import F1Score

        # Creating metric
        self.f1 = F1Score(task="binary").to(config.device)

    @torch.no_grad()
    def key_initialization_epoch(
        self, model: TorchModel, train_data_source: TorchDataSource
    ):

        for batch in tqdm(train_data_source):

            images, _ = batch
            images = images.to(config.device)

            _ = model(images)

    @torch.no_grad()
    def val_epoch(
        self,
        model: TorchModel,
        val_dataset: Dataset,
        crop_size: Tuple[int],
    ):

        model.on_val_epoch_start()

        # Epoch validation
        loss, f1 = evaluate_dataset_patchwise(
            model=model,
            dataset=val_dataset,
            crop_size=crop_size,
        )

        self._update_history_val(loss=loss, f1=f1)

    def _update_batch_history(
        self,
        result: Dict[Any, Any],
        split: str = "train",
    ):
        super()._update_batch_history(result=result, split=split)

        if "batches_f1" in self._history.get(split, []):
            self._history[split]["batches_f1"] = {}

        i_batch = len(self._history[split]["batches_f1"])
        if self._history[split]["batches_f1"].get(i_batch) is None:
            self._history[split]["batches_f1"][i_batch] = []

        self._history[split]["batches_f1"][i_batch].append(
            self.f1(
                result["output"].detach().flatten(),
                result["labels"].detach().flatten().type(torch.int32),
            ).item()
        )

    def _update_history_val(self, loss: torch.Tensor, f1: torch.Tensor, **kwargs):

        if "loss" in self._history.get("val", []):
            self._history["val"]["loss"] = []

        if "f1" in self._history.get("val", []):
            self._history["val"]["f1"] = []

        self._history["val"]["loss"].append(loss)
        self._history["val"]["f1"].append(f1)
