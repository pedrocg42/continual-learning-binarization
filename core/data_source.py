from abc import ABC
from torch.utils.data import DataLoader, Dataset


class DataSource(ABC):
    pass


class TorchDataSource(ABC):
    def __init__(self, dataset: Dataset):

        self.__dataset = dataset
