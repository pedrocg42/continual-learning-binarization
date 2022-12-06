from typing import Tuple

import config
import imagesize
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score, MeanSquaredError
from tqdm import tqdm


@torch.no_grad()
def evaluate_dataset_patchwise(
    model: nn.Module,
    data_loader: DataLoader,
    crop_size: Tuple[int] = (256, 256),
):

    # Initializing metrics
    f1 = F1Score(num_classes=1, task="binary")
    mse = MeanSquaredError()

    pbar = tqdm(data_loader)

    tp = 0
    fp = 0
    fn = 0
    mse = 0.0
    num_pixels = 0
    for i, (patches, label) in enumerate(pbar):

        patches = patches[0].to(config.device)
        label = label[0]

        # Inference
        output_patches = model(patches)

        height, width = label.shape[-2:]

        output_image = torch.zeros((1, height, width), dtype=torch.float32)

        # Calculating necessary patches
        num_patches_height = np.ceil(height / crop_size[0]).astype(int)
        num_patches_width = np.ceil(width / crop_size[1]).astype(int)

        k = 0
        for i in range(num_patches_height):
            for j in range(num_patches_width):

                top = i * crop_size[0]
                left = j * crop_size[1]

                # Avoiding the patch to be outside the image
                if top + crop_size[0] >= height:
                    top = height - crop_size[0]
                if left + crop_size[1] >= width:
                    left = width - crop_size[1]

                output_image[:, top : top + crop_size[0], left : left + crop_size[1]] = output_patches[k]
                k += 1

        # Extracting metrics
        num_pixels += output_image.flatten().shape[0]
        mse += torch.sum((output_image - label) ** 2)
        tp += torch.count_nonzero(torch.logical_and(output_image.flatten() >= 0.5, label.flatten() == 1)).item()
        fp += torch.count_nonzero(torch.logical_and(output_image.flatten() >= 0.5, label.flatten() == 0)).item()
        fn += torch.count_nonzero(torch.logical_and(output_image.flatten() < 0.5, label.flatten() == 1)).item()

    # Calculating MSE
    mse = mse.item() / num_pixels

    # Calculating F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return mse, f1
