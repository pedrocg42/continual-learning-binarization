from typing import Dict
from abc import ABC, abstractmethod


class Callback(ABC):
    def __call__(self, history: Dict):
        return self.should_stop(history)

    @abstractmethod
    def should_stop(history):
        pass


class NumberOfEpochsTrainStopper(Callback):
    def __init__(self, max_epochs: int):

        self.__max_epochs = max_epochs

    def should_stop(self, history: Dict):
        return history.get("num_epochs", 0) > self.__max_epochs


class EarlyStopping(Callback):
    def __init__(self, max_epochs: int):

        self.__max_epochs = max_epochs

    def should_stop(self, history: Dict):
        return history.get("num_epochs", 0) > self.__max_epochs


##################################################################
##############                  META                ##############
##################################################################


class MetaTrainStopper(Callback):
    def __init__(self, max_epochs: int):

        self.__max_epochs = max_epochs

    def should_stop(self, history: Dict):
        return history.get("num_epochs", 0) > self.__max_epochs
