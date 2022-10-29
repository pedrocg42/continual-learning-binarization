import os
from typing import List, Tuple, Union

import numpy as np
import fire
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchmetrics import F1Score
from tqdm import tqdm

import config
from my_utils.evaluate import evaluate_dataset_patchwise
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
    patience: int,
    num_epochs_initialization_keys: int = None,
    device: str = DEVICE,
    force_train: bool = False,
    **experiment,
):

    print(f"Training experiment: {experiment['name']}")

    # Building the model
    print(" > Loading model:")
    model = architecture(**experiment)
    model.to(device)
    print(model)
    print(f"Encoder total parameters: {sum(param.numel() for param in model.parameters())}")

    # Creating metric
    f1 = F1Score().to(device)

    for i_dataset, dataset_group in enumerate(datasets):

        if i_dataset == 0:
            model_name = experiment["name"]
            model_name += f"__{'_'.join(dataset_group)}"
            model_file_path = os.path.join(config.models_path, f"{model_name}.pt")
            last_model_file_path = model_file_path
        else:
            last_model_file_path = model_file_path
            model_name += f"__{'_'.join(dataset_group)}"
            model_file_path = os.path.join(config.models_path, f"{model_name}.pt")

        # Checking of model already trained exists
        if os.path.exists(model_file_path) and not force_train:
            continue
        
        print(f" > Training model {model_name}")
        # Loading best model from last training if it is not the first one
        if i_dataset > 0:
            # Freezing decoder after training with the first set
            if experiment["architecture_type"] == "discrete_key_value_bottleneck":
                experiment["freeze_decoder"] = True
                model = architecture(**experiment)
                model.to(device)

            print(f" > Loading model weights from {last_model_file_path}")
            model.load_state_dict(torch.load(last_model_file_path))

            # Configurating model to only train certaing parts of it
            if experiment["architecture_type"] == "discrete_key_value_bottleneck":
                # All model frozen but values
                model.train(False)
            elif experiment["architecture_type"] == "vector_quantized":
                # Freezing Keys and encoder
                model.train(False)
                model.decoder.train(True)
            elif experiment["architecture_type"] == "baseline":
                # Freezing Keys and encoder
                model.train(False)
                model.decoder.train(True)

        # Preparing train dataset
        print(f" > Creating Training Dataset for {dataset_group}")
        train_dataset = dataset(
            datasets=dataset_group,
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
            num_workers=4,
            pin_memory=True,
        )

        # Preparing validation dataset
        print(" > Creating Validation Dataset")
        val_dataset = dataset(
            datasets=dataset_group,
            train_val_test_split=train_val_test_split,
            split="val",
            crop_size=crop_size,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        # Preparing images to check progress
        images_show = torch.zeros((16, 3, 256, 256))
        labels_show = torch.zeros((16, 1, 256, 256))
        for i, idx in enumerate(np.arange(0, len(val_dataset) - len(val_dataset) // 16, len(val_dataset) // 16)):
            image, label = val_dataset[idx]

            images_show[i] = torch.unsqueeze(image, 0)
            labels_show[i] = torch.unsqueeze(label, 0)

        # Configure optimizer and loss function
        optimizer_ = optimizer(model.parameters(), lr=learning_rate)
        criteria_ = criteria()

        # Initialiazing Tensorboard logging and adding model graph
        writer = SummaryWriter(log_dir=os.path.join(config.logs_path, model_name))
        # writer.add_graph(model, torch.zeros((8, 3, 256, 256), dtype=torch.float32).to(device))

        # Keys initialization for descrete key-value bottleneck
        if i_dataset == 0 and experiment["architecture_type"] == "discrete_key_value_bottleneck":
            print("[PHASE-0] Keys Initialization:")

            model.train()
            model.encoder.train(False)

            # Start Training
            with torch.no_grad():
                for epoch in range(num_epochs_initialization_keys):

                    print(f" > Training epoch {epoch + 1} of {num_epochs_initialization_keys}")

                    # Epoch training
                    pbar = tqdm(train_dataloader)
                    for step, batch in enumerate(pbar):

                        images, labels = batch
                        images = images.to(device)

                        # Inference
                        output = model(images)

        print("[PHASE-1] Training Model:")
        best_val_f1 = 0.0
        patience_iterations = 0
        for epoch in range(num_epochs):

            print(f" > Training epoch {epoch + 1} of {num_epochs}")

            model.train()

            if "dkvb" in experiment["name"]:
                # Freezing encoder and keys
                model.encoder.train(False)
                model.key_value_bottleneck.vq.train(False)

            # Epoch training
            running_loss = 0.0
            running_f1 = 0.0
            pbar = tqdm(train_dataloader)
            for step, batch in enumerate(pbar):

                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                # Zero gradient before every batch
                optimizer_.zero_grad()

                # Inference
                output = model(images)

                # Compute loss
                loss = criteria_(output, labels)
                loss.backward()

                # Adjust weights
                optimizer_.step()

                # Computing training mean loss and f1 score
                running_loss += loss.item()
                avg_loss = running_loss / (step + 1)  # add batch_size
                running_f1 += f1(output.detach().flatten(), labels.detach().flatten().type(torch.int32)).item()
                avg_f1 = running_f1 / (step + 1)  # add batch_size

                pbar.set_postfix({"loss": avg_loss, "f1": avg_f1})

            # Adding logs for every epoch
            writer.add_scalar("Train Loss", avg_loss, epoch)
            writer.add_scalar("Train F1 Score", avg_f1, epoch)

            print(f" > Evaluating validation")

            model.eval()

            # Epoch validation
            val_loss, val_f1 = evaluate_dataset_patchwise(
                model=model, dataset=val_dataset, data_loader=val_dataloader, crop_size=crop_size, device=device
            )

            # Adding logs for every epoch
            writer.add_scalar("Validation Loss", avg_loss, epoch)
            writer.add_scalar("Validation F1 Score", val_f1, epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_iterations = 0

                torch.save(model.state_dict(), model_file_path)

                print(f" > New best model found with best validation loss {val_loss} and F1-Score {val_f1}")
                print(f" > New best model saved in {model_file_path}")

            else:
                patience_iterations += 1
                if patience_iterations >= patience:
                    break

            # Saving images to check progress
            output_show = model(images_show.to(device))
            grid = make_grid(
                torch.cat(
                    [
                        images_show.detach().cpu()[:16],
                        torch.tile(labels_show.detach().cpu()[:16], (1, 3, 1, 1)),
                        torch.tile(output_show.detach().cpu()[:16], (1, 3, 1, 1)),
                    ]
                )
            )
            writer.add_image(f"{model_name}/images_epoch_{epoch+1}", grid, epoch)

        writer.close()
        print("End")


if __name__ == "__main__":
    fire.Fire(train)
