# Architectures
from networks.dkvb import DKVBBin
from networks.unet import UNet
from networks.baseline import Baseline
from networks.vq_bin import VQBin

# Optimizers
from torch.optim import Adam

# Losses
from torch.nn import MSELoss

# Datasets
from data import BinDataset


unet_all_sets = {
    "name": "unet_all_sets",
    # Model
    "architecture": UNet,
    "hidden_dims_encoder": [16, 32, 64, 128, 256],
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}

baseline_all_sets = {
    "name": "baseline_all_sets",
    # Model
    "architecture": Baseline,
    "hidden_dims_encoder": [16, 32, 64, 128, 256],
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}


# Baselines Vector Quantizer Experiments
vq_all_sets = {
    "name": "vq_all_sets",
    # Model
    "architecture": VQBin,
    "hidden_dims_encoder": [16, 32, 64, 128, 256],
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    "codebook_size": 8192,
    "num_codebooks": 1,
    "decay": 0.90,
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}

vq_all_sets_64 = {
    "name": "vq_all_sets_64",
    # Model
    "architecture": VQBin,
    "hidden_dims_encoder": [16, 32, 64, 128, 256],
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    "codebook_size": 128,
    "num_codebooks": 64,
    "decay": 0.90,
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}

vq_all_sets_128 = {
    "name": "vq_all_sets_128",
    # Model
    "architecture": VQBin,
    "hidden_dims_encoder": [16, 32, 64, 128, 256],
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    "codebook_size": 64,
    "num_codebooks": 128,
    "decay": 0.90,
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}

vq_all_sets_512 = {
    "name": "vq_all_sets_512",
    # Model
    "architecture": VQBin,
    "hidden_dims_encoder": [16, 32, 64, 128, 256],
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    "codebook_size": 16,
    "num_codebooks": 512,
    "decay": 0.90,
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}

# Baselines Discrete Key-Value Bottleneck Experiments
dkvb_all_sets = {
    "name": "dkvb_all_sets",
    # Model
    "architecture": DKVBBin,
    "encoder_model_experiment": baseline_all_sets,
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    "codebook_size": 8192,
    "num_codebooks": 1,
    "decay": 0.90,
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs_initialization_keys": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}

dkvb_all_sets_64 = {
    "name": "dkvb_all_sets_64",
    # Model
    "architecture": DKVBBin,
    "encoder_model_experiment": baseline_all_sets,
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    "codebook_size": 128,
    "num_codebooks": 64,
    "decay": 0.90,
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs_initialization_keys": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}

dkvb_all_sets_128 = {
    "name": "dkvb_all_sets_128",
    # Model
    "architecture": DKVBBin,
    "encoder_model_experiment": baseline_all_sets,
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    "codebook_size": 64,
    "num_codebooks": 128,
    "decay": 0.90,
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs_initialization_keys": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}

dkvb_all_sets_512 = {
    "name": "dkvb_all_sets_512",
    # Model
    "architecture": DKVBBin,
    "encoder_model_experiment": baseline_all_sets,
    "hidden_dims_decoder": [256, 128, 64, 32, 16],
    "codebook_size": 16,
    "num_codebooks": 512,
    "decay": 0.90,
    # Dataset
    "dataset": BinDataset,
    "datasets": ["Dibco", "Einsieldeln", "Palm", "PHI", "Salzinnes"],
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs_initialization_keys": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "steps_per_epoch": 50,
}
