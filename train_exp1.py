import os
from typing import List, Tuple, Union

import fire
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchmetrics import F1Score
from tqdm import tqdm

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
    criteria: nn.Module,
    optimizer: nn.Module,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    patience: int = 10000,
    device: str = DEVICE,
    **experiment,
):

    print(f"Training experiment: {experiment['name']}")

    # Building the model
    print(" > Loading model:")
    model = architecture(**experiment)
    model.to(device)
    print(model)
    print(f"Encoder total parameters: {sum(param.numel() for param in model.parameters())}")

    # Preparing train dataset
    print(" > Creating Training Dataset")
    train_dataset = dataset(
        datasets=datasets,
        train_val_test_split=train_val_test_split,
        split="train",
        crop_size=crop_size,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Preparing validation dataset
    print(" > Creating Validation Dataset")
    val_dataset = dataset(
        datasets=datasets,
        train_val_test_split=train_val_test_split,
        split="val",
        crop_size=crop_size,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )

    # Configure optimizer and loss function
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    criteria = criteria()

    # Initialiazing Tensorboard logging and adding model graph
    writer = SummaryWriter(log_dir=os.path.join(config.logs_path, experiment["name"]))
    writer.add_graph(model, torch.zeros((100, 3, 32, 32), dtype=torch.float32).to(device))
    model_file_path = os.path.join(config.models_path, f"{experiment['name']}.pt")

    # Initializing metrics
    f1 = F1Score(num_classes=1).to(device)
    best_f1 = 0.0
    patience_iterations = 0

    for epoch in range(num_epochs):

        model.train()

        # Epoch training
        running_loss = 0.0
        running_f1 = 0.0
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Zero gradient before every batch
            optimizer.zero_grad()

            # Inference
            output = model(images)

            # Compute loss
            loss = criteria(output, labels)
            loss.backward()

            # Adjust weights
            optimizer.step()

            # Computing training mean loss and f1 score
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)  # add batch_size
            running_f1 += f1(output.detach().flatten(), labels.detach().flatten().type(torch.int32)).item()
            avg_f1 = running_f1 / (i + 1)  # add batch_size

            pbar.set_postfix({"loss": avg_loss, "f1": avg_f1})

        # Adding logs for every epoch
        writer.add_scalar("Train Loss", avg_loss, epoch)
        writer.add_scalar("Train F1 Score", avg_f1, epoch)

        model.eval()

        # Epoch training
        running_loss = 0.0
        running_f1 = 0.0
        pbar = tqdm(val_dataloader)
        for i, batch in enumerate(pbar):

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Inference
            output = model(images)

            # Compute loss
            loss = criteria(output, labels)

            # Computing training mean loss and f1 score
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)  # add batch_size
            running_f1 += f1(output.detach().flatten(), labels.detach().flatten().type(torch.int32)).item()
            avg_f1 = running_f1 / (i + 1)  # add batch_size

            pbar.set_postfix({"loss": avg_loss, "f1": avg_f1})

        # Adding logs for every epoch
        writer.add_scalar("Validation Loss", avg_loss, epoch)
        writer.add_scalar("Validation F1 Score", avg_f1, epoch)

        # Saving images to check progress
        grid = make_grid(torch.cat([labels.detach().cpu()[:32], output.detach().cpu()[:32]]))
        writer.add_image(f"{experiment['experiment']}/images_epoch_{epoch+1}", grid, epoch)
        grid = make_grid(torch.stack((output[:32], labels[:32]), dim=0).view(64, 1, 256, 256))
        writer.add_image(f"{experiment['experiment']}/images_epoch_{epoch+1}_2", grid, epoch)

        if avg_f1 < best_f1:
            best_f1 = avg_f1
            best_model_epoch = epoch
            patience_iterations = 0

            model_file_path = os.path.join(config.model_path, experiment["experiment"])
            torch.save(model.state_dict(), model_file_path)

            print(f" > New best model found with best validation loss: {avg_f1}")
            print(f" > New best model saved in {model_file_path}")

        else:
            patience_iterations += 1
            if patience_iterations >= patience:
                break

    writer.close()
    print("End")


if __name__ == "__main__":
    fire.Fire(train)
