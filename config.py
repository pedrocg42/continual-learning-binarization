import os

seed = 42

device = "cuda"

dataset_path = "data/"
cross_val_splits = 5

models_path = "results/models"
os.makedirs(models_path, exist_ok=True)

logs_path = "results/logs"
os.makedirs(logs_path, exist_ok=True)

ALL_DATASETS = [["PHI"], ["Salzinnes"], ["Dibco"], ["Einsieldeln"]]  # ["Palm"]
