# Architectures
from networks.discrete_key_value_bottleneck import DKVBBin

# Optimizers
from torch.optim import Adam

# Losses
from torch.nn import MSELoss

# Datasets
from data import BinDataset


# Continual learning experiments:
# These sequences are not arbitrary but based on the differences
# between documents: scenarios 1 and 2 alternate musicand text documents,
# while scenarios 3 and 4 group tasks with similar characteristics by pairs.
# Note that the latter ones can be considered more challenging because of
# having to learn two tasks once the other two tasks, with very different
# characteristics, have been learned.

scenario_1 = [["PHI"], ["Salzinnes"], ["Dibco"], ["Einsieldeln"]]
scenario_2 = [["Einsieldeln"], ["Dibco"], ["Salzinnes"], ["PHI"]]
scenario_3 = [["PHI"], ["Dibco"], ["Einsieldeln"], ["Salzinnes"]]
scenario_4 = [["Salzinnes"], ["Einsieldeln"], ["Dibco"], ["PHI"]]

#################################################################
################            SCENARIO 1              #############
#################################################################

baseline_scenario_1 = {
    "name": "baseline_scenario_1",
    # Model
    "architecture": DKVBBin,
    "architecture_type": "baseline",
    "pretrained_encoder": "dino_resnet_50",
    "embedding_dim": 1024,
    "hidden_dims_decoder": [256, 128, 64, 32],
    # Dataset
    "dataset": BinDataset,
    "datasets": scenario_1,
    "train_val_test_split": [0.6, 0.2],
    "crop_size": (256, 256),
    # Training
    "criteria": MSELoss,
    "optimizer": Adam,
    "learning_rate": 3e-4,
    "num_epochs": 400,
    "batch_size": 32,
    "steps_per_epoch": 50,
    "patience": 25,
    "patience_learning_rate": 5,
}

# Baselines Vector Quantizer Experiments
vq_scenario_1 = baseline_scenario_1.copy()
vq_scenario_1.update(
    {
        "name": "vq_scenario_1",
        "architecture_type": "vector_quantizer",
        "codebook_size": 8192,
        "num_codebooks": 1,
        "vq_decay": 0.9,
        "threshold_ema_dead_code": 1024,
        "num_epochs_initialization_keys": 10,
    }
)

vq_scenario_1_64 = vq_scenario_1.copy()
vq_scenario_1_64.update(
    {
        "name": "vq_scenario_1_64",
        "codebook_size": 128,
        "num_codebooks": 64,
    }
)

vq_scenario_1_128 = vq_scenario_1.copy()
vq_scenario_1_128.update(
    {
        "name": "vq_scenario_1_128",
        "codebook_size": 64,
        "num_codebooks": 128,
    }
)

vq_scenario_1_512 = vq_scenario_1.copy()
vq_scenario_1_512.update(
    {
        "name": "vq_scenario_1_512",
        "codebook_size": 16,
        "num_codebooks": 512,
    }
)

# Baselines Discrete Key-Value Bottleneck Experiments
dkvb_scenario_1 = baseline_scenario_1.copy()
dkvb_scenario_1.update(
    {
        "name": "dkvb_scenario_1",
        "architecture_type": "discrete_key_value_bottleneck",
        "codebook_size": 8192,
        "num_codebooks": 1,
        "vq_decay": 0.9,
        "threshold_ema_dead_code": 1024,
        "num_epochs_initialization_keys": 10,
    }
)

dkvb_scenario_1_64 = dkvb_scenario_1.copy()
dkvb_scenario_1_64.update(
    {
        "name": "dkvb_scenario_1_64",
        "codebook_size": 128,
        "num_codebooks": 64,
    }
)

dkvb_scenario_1_128 = dkvb_scenario_1.copy()
dkvb_scenario_1_128.update(
    {
        "name": "dkvb_scenario_1_128",
        "codebook_size": 64,
        "num_codebooks": 128,
    }
)

dkvb_scenario_1_512 = dkvb_scenario_1.copy()
dkvb_scenario_1_512.update(
    {
        "name": "dkvb_scenario_1_512",
        "codebook_size": 16,
        "num_codebooks": 512,
    }
)

#################################################################
################            SCENARIO 2              #############
#################################################################

baseline_scenario_2 = baseline_scenario_1.copy()
baseline_scenario_2.update(
    {
        "name": "baseline_scenario_2",
        "datasets": scenario_2,
    }
)

# Baselines Vector Quantizer Experiments
vq_scenario_2 = baseline_scenario_2.copy()
vq_scenario_2.update(
    {
        "name": "vq_scenario_2",
        "architecture_type": "vector_quantizer",
        "codebook_size": 8192,
        "num_codebooks": 1,
        "vq_decay": 0.9,
        "threshold_ema_dead_code": 1024,
        "num_epochs_initialization_keys": 10,
    }
)

vq_scenario_2_64 = vq_scenario_2.copy()
vq_scenario_2_64.update(
    {
        "name": "vq_scenario_2_64",
        "codebook_size": 128,
        "num_codebooks": 64,
    }
)

vq_scenario_2_128 = vq_scenario_2.copy()
vq_scenario_2_128.update(
    {
        "name": "vq_scenario_2_128",
        "codebook_size": 64,
        "num_codebooks": 128,
    }
)

vq_scenario_2_512 = vq_scenario_2.copy()
vq_scenario_2_512.update(
    {
        "name": "vq_scenario_2_512",
        "codebook_size": 16,
        "num_codebooks": 512,
    }
)

# Baselines Discrete Key-Value Bottleneck Experiments
dkvb_scenario_2 = baseline_scenario_2.copy()
dkvb_scenario_2.update(
    {
        "name": "dkvb_scenario_2",
        "architecture_type": "discrete_key_value_bottleneck",
        "codebook_size": 8192,
        "num_codebooks": 1,
        "vq_decay": 0.9,
        "threshold_ema_dead_code": 1024,
        "num_epochs_initialization_keys": 10,
    }
)

dkvb_scenario_2_64 = dkvb_scenario_2.copy()
dkvb_scenario_2_64.update(
    {
        "name": "dkvb_scenario_2_64",
        "codebook_size": 128,
        "num_codebooks": 64,
    }
)

dkvb_scenario_2_128 = dkvb_scenario_2.copy()
dkvb_scenario_2_128.update(
    {
        "name": "dkvb_scenario_2_128",
        "codebook_size": 64,
        "num_codebooks": 128,
    }
)

dkvb_scenario_2_512 = dkvb_scenario_2.copy()
dkvb_scenario_2_512.update(
    {
        "name": "dkvb_scenario_1_512",
        "codebook_size": 16,
        "num_codebooks": 512,
    }
)


#################################################################
################            SCENARIO 3              #############
#################################################################

baseline_scenario_3 = baseline_scenario_1.copy()
baseline_scenario_3.update(
    {
        "name": "baseline_scenario_3",
        "datasets": scenario_3,
    }
)

# Baselines Vector Quantizer Experiments
vq_scenario_3 = baseline_scenario_3.copy()
vq_scenario_3.update(
    {
        "name": "vq_scenario_3",
        "architecture_type": "vector_quantizer",
        "codebook_size": 8192,
        "num_codebooks": 1,
        "vq_decay": 0.9,
        "threshold_ema_dead_code": 1024,
        "num_epochs_initialization_keys": 10,
    }
)

vq_scenario_3_64 = vq_scenario_3.copy()
vq_scenario_3_64.update(
    {
        "name": "vq_scenario_3_64",
        "codebook_size": 128,
        "num_codebooks": 64,
    }
)

vq_scenario_3_128 = vq_scenario_3.copy()
vq_scenario_3_128.update(
    {
        "name": "vq_scenario_3_128",
        "codebook_size": 64,
        "num_codebooks": 128,
    }
)

vq_scenario_3_512 = vq_scenario_3.copy()
vq_scenario_3_512.update(
    {
        "name": "vq_scenario_3_512",
        "codebook_size": 16,
        "num_codebooks": 512,
    }
)

# Baselines Discrete Key-Value Bottleneck Experiments
dkvb_scenario_3 = baseline_scenario_3.copy()
dkvb_scenario_3.update(
    {
        "name": "dkvb_scenario_3",
        "architecture_type": "discrete_key_value_bottleneck",
        "codebook_size": 8192,
        "num_codebooks": 1,
        "vq_decay": 0.9,
        "threshold_ema_dead_code": 1024,
        "num_epochs_initialization_keys": 10,
    }
)

dkvb_scenario_3_64 = dkvb_scenario_3.copy()
dkvb_scenario_3_64.update(
    {
        "name": "dkvb_scenario_3_64",
        "codebook_size": 128,
        "num_codebooks": 64,
    }
)

dkvb_scenario_3_128 = dkvb_scenario_3.copy()
dkvb_scenario_3_128.update(
    {
        "name": "dkvb_scenario_3_128",
        "codebook_size": 64,
        "num_codebooks": 128,
    }
)

dkvb_scenario_3_512 = dkvb_scenario_3.copy()
dkvb_scenario_3_512.update(
    {
        "name": "dkvb_scenario_3_512",
        "codebook_size": 16,
        "num_codebooks": 512,
    }
)


#################################################################
################            SCENARIO 4              #############
#################################################################

baseline_scenario_4 = baseline_scenario_1.copy()
baseline_scenario_4.update(
    {
        "name": "baseline_scenario_4",
        "datasets": scenario_4,
    }
)

# Baselines Vector Quantizer Experiments
vq_scenario_4 = baseline_scenario_4.copy()
vq_scenario_4.update(
    {
        "name": "vq_scenario_4",
        "architecture_type": "vector_quantizer",
        "codebook_size": 8192,
        "num_codebooks": 1,
        "vq_decay": 0.9,
        "threshold_ema_dead_code": 1024,
        "num_epochs_initialization_keys": 10,
    }
)

vq_scenario_4_64 = vq_scenario_4.copy()
vq_scenario_4_64.update(
    {
        "name": "vq_scenario_4_64",
        "codebook_size": 128,
        "num_codebooks": 64,
    }
)

vq_scenario_4_128 = vq_scenario_4.copy()
vq_scenario_4_128.update(
    {
        "name": "vq_scenario_4_128",
        "codebook_size": 64,
        "num_codebooks": 128,
    }
)

vq_scenario_4_512 = vq_scenario_4.copy()
vq_scenario_4_512.update(
    {
        "name": "vq_scenario_4_512",
        "codebook_size": 16,
        "num_codebooks": 512,
    }
)

# Baselines Discrete Key-Value Bottleneck Experiments
dkvb_scenario_4 = baseline_scenario_4.copy()
dkvb_scenario_4.update(
    {
        "name": "dkvb_scenario_4",
        "architecture_type": "discrete_key_value_bottleneck",
        "codebook_size": 8192,
        "num_codebooks": 1,
        "vq_decay": 0.9,
        "threshold_ema_dead_code": 1024,
        "num_epochs_initialization_keys": 10,
    }
)

dkvb_scenario_4_64 = dkvb_scenario_4.copy()
dkvb_scenario_4_64.update(
    {
        "name": "dkvb_scenario_4_64",
        "codebook_size": 128,
        "num_codebooks": 64,
    }
)

dkvb_scenario_4_128 = dkvb_scenario_4.copy()
dkvb_scenario_4_128.update(
    {
        "name": "dkvb_scenario_4_128",
        "codebook_size": 64,
        "num_codebooks": 128,
    }
)

dkvb_scenario_4_512 = dkvb_scenario_4.copy()
dkvb_scenario_4_512.update(
    {
        "name": "dkvb_scenario_4_512",
        "codebook_size": 16,
        "num_codebooks": 512,
    }
)
