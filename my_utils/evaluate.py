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
    dataset: DataLoader,
    crop_size: Tuple[int] = (256, 256),
    batch_size: int = 32,
):

    # Initializing metrics
    f1 = F1Score(num_classes=1, task="binary")
    mse = MeanSquaredError()

    pbar = tqdm(dataset)

    tp = 0
    fp = 0
    fn = 0
    mse = 0.0
    num_pixels = 0
    for i, (patches, label) in enumerate(pbar):

        patches = patches.to(config.device)
        label = label

        output_list = []
        for j in range(0, len(patches), batch_size):
            # Inference
            output_list.append(model(patches[j : j + batch_size]))

        output_patches = torch.cat(output_list)

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
    precision = tp / (tp + fp + torch.finfo(torch.float32).eps)
    recall = tp / (tp + fn + torch.finfo(torch.float32).eps)
    f1 = (2 * precision * recall) / (precision + recall + torch.finfo(torch.float32).eps)

    return mse, f1
