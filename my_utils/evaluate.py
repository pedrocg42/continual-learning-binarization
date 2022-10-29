from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score, MeanSquaredError
from tqdm import tqdm


def evaluate_dataset_patchwise(
    model: nn.Module,
    dataset: Dataset,
    data_loader: DataLoader,
    crop_size: Tuple[int] = (256, 256),
    device: str = "cuda",
):

    # Initializing metrics
    f1 = F1Score(num_classes=1)
    mse = MeanSquaredError()

    # Calculating number of patches per image
    num_patches = [
        np.ceil(height / crop_size[0]).astype(int) * np.ceil(width / crop_size[1]).astype(int)
        for width, height in dataset.images_sizes
    ]

    with torch.no_grad():

        image_index = 0
        label_queue_patches = torch.empty((0, 1, *crop_size))
        output_queue_patches = torch.empty((0, 1, *crop_size))
        pbar = tqdm(data_loader)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        mse = 0.0
        num_pixels = 0
        for i, batch in enumerate(pbar):

            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            # Inference
            output = model(images)

            # Concatenatig new patches to queue
            label_queue_patches = torch.cat((label_queue_patches, labels.cpu()))
            output_queue_patches = torch.cat((output_queue_patches, output.cpu()))

            # Checking if there's enough patches to form an image
            if label_queue_patches.shape[0] >= num_patches[image_index]:
                # Ensambling as many images as the patches extracted allow us
                while label_queue_patches.shape[0] >= num_patches[image_index]:

                    width, height = dataset.images_sizes[image_index]

                    output_image = torch.zeros((1, height, width), dtype=torch.float32)
                    label_image = torch.zeros((1, height, width), dtype=torch.float32)

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

                            output_image[
                                :, top : top + crop_size[0], left : left + crop_size[1]
                            ] = output_queue_patches[k]
                            label_image[:, top : top + crop_size[0], left : left + crop_size[1]] = label_queue_patches[
                                k
                            ]
                            k += 1

                    # Deleting used patches
                    label_queue_patches = label_queue_patches[k:]
                    output_queue_patches = output_queue_patches[k:]

                    # Extracting metrics
                    num_pixels += output_image.flatten().shape[0]
                    mse += torch.sum((output_image - label_image) ** 2)
                    tp += torch.count_nonzero(
                        torch.logical_and(output_image.flatten() >= 0.5, label_image.flatten() == 1)
                    ).item()
                    # tn += torch.count_nonzero(
                    #     torch.logical_and(output_image.flatten() < 0.5, label_image.flatten() == 0)
                    # ).item()
                    fp += torch.count_nonzero(
                        torch.logical_and(output_image.flatten() >= 0.5, label_image.flatten() == 0)
                    ).item()
                    fn += torch.count_nonzero(
                        torch.logical_and(output_image.flatten() < 0.5, label_image.flatten() == 1)
                    ).item()

    # Calculating MSE
    mse = mse.item() / num_pixels

    # Calculating F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return mse, f1
