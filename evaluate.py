import os
from this import d
from typing import List, Tuple, Union

import numpy as np
import fire
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score, MeanSquaredError
from tqdm import tqdm
import csv

import config
from utils_experiments import parse_experiment

DEVICE = "cuda"


@parse_experiment
def train(
    architecture,
    dataset: Dataset,
    datasets: Union[List[str], str],
    train_val_test_split: List[float],
    crop_size: Tuple[int],
    batch_size: int,
    steps_per_epoch: int,
    device: str = DEVICE,
    **experiment,
):

    print(f"Training experiment: {experiment['name']}")

    # Building the model
    print("Building architecture")
    model = architecture(**experiment)
    model.to(device)
    print(model)
    print(f"Encoder total parameters: {sum(param.numel() for param in model.parameters())}")

    model_file_path = os.path.join(config.models_path, f"{experiment['name']}.pt")
    print(f" > Loading model from {model_file_path}")
    model.load_state_dict(torch.load(model_file_path))

    # Preparing testing dataset
    print(" > Creating Validation Dataset")
    test_dataset = dataset(
        datasets=datasets,
        train_val_test_split=train_val_test_split,
        split="test",
        crop_size=crop_size,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Initializing metrics
    f1 = F1Score(num_classes=1)
    mse = MeanSquaredError()

    model.eval()

    # Calculating number of patches per image
    num_patches = [
        np.ceil(height / crop_size[0]).astype(int) * np.ceil(width / crop_size[1]).astype(int)
        for width, height in test_dataset.images_sizes
    ]

    with torch.no_grad():
        # Epoch training
        mses = []
        f1s = []
        image_index = 0
        label_queue_patches = torch.empty((0, 1, *crop_size))
        output_queue_patches = torch.empty((0, 1, *crop_size))
        pbar = tqdm(test_dataloader)

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

                    width, height = test_dataset.images_sizes[image_index]

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
                    mses.append(mse(output_image, label_image))
                    f1s.append(f1(output_image.flatten(), label_image.flatten().type(torch.int32)))

    # Extracting metrics
    print(f" > Saving metrics...")
    csv_path = "results.csv"
    existed = os.path.exists(csv_path)

    f = open(csv_path, "a")
    csv_writer = csv.writer(f, delimiter="\t")

    if not existed:
        # Creating and writing header
        header = ["Experiment", "MSE", "F1 Score"]
        csv_writer.writerow(header)

    # Evaluating overall and for every single class
    row_results = [experiment["name"], np.mean(mses), np.mean(f1s)]

    csv_writer.writerow(row_results)
    print(f" > Results saved in {csv_path}")

    # Close file
    f.close()

    print("End")


if __name__ == "__main__":
    fire.Fire(train)
