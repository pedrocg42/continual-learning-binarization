import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

import config
from core.data_source import DataSource, TorchDataSource
from core.model import Model, TorchModel
from core.train_looper import TorchLooper


class Trainer(ABC):
    def __init__(
        self,
        learning_rate=3e-4,
        steps_per_epoch: Union[float, int] = 1.0,
        num_epochs: int = 100,
        patience: int = None,
        patience_learning_rate: int = None,
        **experiment,
    ):
        self._learning_rate = learning_rate
        self._steps_per_epoch = steps_per_epoch
        self._num_epochs = num_epochs
        self._patience = patience
        self._patience_learning_rate = patience_learning_rate

    def build(self, architecture: Model, **experiment):

        self._model = architecture(**experiment)

    def __call__(self, **kwargs):
        return self._train(**kwargs)

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass


class TorchTrainer(Trainer):
    def __init__(
        self,
        train_looper: TorchLooper,
        criteria: _Loss,
        optimizer=Optimizer,
        **experiment,
    ):
        super().__init__(**experiment)

        self._train_looper = train_looper
        self._criteria = criteria
        self._optimizer = optimizer

    def build(self, architecture: nn.Module, **experiment):

        logging.info(" > Building model")
        self._model = architecture(**experiment)
        self._model.to(config.device)
        logging.info(self._model)
        logging.info(
            f"> > Total parameters: {sum(param.numel() for param in self._model.parameters())}"
        )

        self._optimizer_ = self._optimizer(
            self._model.parameters(), lr=self._learning_rate
        )

    def _train(self, **experiment):

        for epoch in range(self._num_epochs):

            logging.info(f"Training epoch {epoch} of {self._num_epochs}")

            self._train_looper.train_epoch(
                model=self._model,
                optimizer=self._optimizer,
                criteria=self._criteria,
                train_data_source=train_data_source,
            )

            if val_data_source is not None:
                self._train_looper.val_epoch(
                    model=self._model,
                    optimizer=self._optimizer,
                    criteria=self._criteria,
                    val_data_source=val_data_source,
                )


class ClBinarizationTorchTrainer(TorchTrainer):
    def __init__(
        self,
        num_epochs_initialization_keys: int,
        **experiment,
    ):
        super().__init__(**experiment)

        self.__num_epochs_initialization_keys = num_epochs_initialization_keys

    def _train(self, name: str, datasets: List[str], **experiment):

        logging.info(f" > Starting training experiment: {name}")

        for i_cross_val in range(config.cross_val_splits):
            logging.info(
                f" > >  Starting cross-validation: {i_cross_val+1} of {config.cross_val_splits}"
            )

            self.build(**experiment)

            for i_dataset, dataset_name in enumerate(datasets):

                self._build_model_path(
                    experiment_name=name,
                    i_cross_val=i_cross_val,
                    i_dataset=i_dataset,
                    dataset_name=dataset_name,
                )

                # Checking of model already trained exists
                if os.path.exists(self.__model_file_path):
                    print(
                        f" > > Found model already trained. Jumping to next dataset..."
                    )
                    continue

                logging.info(
                    f" > > >  Starting training on dataset {i_dataset+1} of {len(datasets)}: {dataset_name}"
                )

                train_data_source, val_data_source = self._build_data_sources(
                    dataset_name=dataset_name, i_cross_val=i_cross_val, **experiment
                )

                if i_dataset == 0:

                    logging.info("[PHASE-0] Keys Initialization:")
                    for epoch in self.__num_epochs_initialization_keys:
                        self._train_looper(
                            model=self._model,
                            optimizer=self._optimizer,
                            criteria=self._criteria,
                            train_data_source=train_data_source,
                        )
                else:
                    logging.info(
                        f" > Loading model weights from {self.__last_model_file_path}"
                    )
                    self._model.load_state_dict(torch.load(self.__last_model_file_path))

                logging.info("[PHASE-1] Training Model:")
                for epoch in range(self._num_epochs):

                    logging.info(f"Training epoch {epoch} of {self._num_epochs}")

                    self._train_looper.train_epoch(
                        model=self._model,
                        optimizer=self._optimizer,
                        criteria=self._criteria,
                        train_data_source=train_data_source,
                    )

                    if val_data_source is not None:
                        self._train_looper.val_epoch(
                            model=self._model,
                            optimizer=self._optimizer,
                            criteria=self._criteria,
                            val_data_source=val_data_source,
                        )

                    # Evaluting epoch results
                    if val_f1 > best_val_f1:
                        # Saving new best model and initialize variables
                        best_val_f1 = val_f1
                        self.__patience_iterations = 0

                        torch.save(self._model.state_dict(), self.__model_file_path)

                        print(
                            f" > New best model found with best F1-Score {val_f1} ({val_loss=})  "
                        )
                        print(f" > New best model saved in {self.__model_file_path}")
                    else:
                        # Reducing learning rate and/or stopping the training
                        self.__patience_iterations += 1
                        if (
                            self._patience_learning_rate is not None
                            and self.__patience_iterations
                            % self._patience_learning_rate
                        ) == 0:
                            self._learning_rate /= 2.0

                        if (
                            self._patience is not None
                            and self.__patience_iterations >= self._patience
                        ):
                            break

    def _build_data_sources(
        dataset: Dataset,
        dataset_name: str,
        train_val_test_split: List[float],
        i_cross_val: int,
        crop_size: Tuple[int],
        batch_size: int,
        steps_per_epoch: int,
        **experiment,
    ):
        # Preparing train dataset
        logging.info(f" > Creating Training Dataset for {dataset_name}")
        train_dataset = dataset(
            datasets=dataset_name,
            train_val_test_split=train_val_test_split,
            split="train",
            crop_size=crop_size,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            cross_val_id=i_cross_val,
        )
        train_data_source = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        # Preparing validation dataset
        logging.info(" > Creating Validation Dataset")
        val_data_source = dataset(
            datasets=dataset_name,
            train_val_test_split=train_val_test_split,
            split="val",
            crop_size=crop_size,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            cross_val_id=i_cross_val,
        )

        return train_data_source, val_data_source

    def _build_model_path(
        self, experiment_name: str, i_cross_val: int, i_dataset: int, dataset_name: str
    ):
        # Building model_name and model_path and checking if it already exists
        if i_dataset == 0:
            model_name = experiment_name + f"_cv_{i_cross_val+1}"
        else:
            self.__last_model_file_path = self.__model_file_path

        model_name += f"__{'_'.join(dataset_name)}"
        self.__model_file_path = os.path.join(config.models_path, f"{model_name}.pt")
