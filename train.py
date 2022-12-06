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
from my_utils.parse_experiment import parse_experiment


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
    patience_learning_rate: int,
    num_epochs_initialization_keys: int = None,
    force_train: bool = False,
    **experiment,
):

    print(f"Training experiment: {experiment['name']}")

    # Creating metric
    f1 = F1Score(task="binary").to(config.device)

    for i_cross_val in range(config.cross_val_splits):

        print(f" > Starting cross-validation: {i_cross_val+1} of {config.cross_val_splits}")

        # Building the model
        print(" > Building model")
        model = architecture(**experiment)
        model.to(config.device)
        print(model)
        print(f"> > Total parameters: {sum(param.numel() for param in model.parameters())}")

        # Configure optimizer and loss function
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        criteria = criteria()

        for i_dataset, dataset_group in enumerate(datasets):

            # Building model_name and model_path and checking if it already exists
            if i_dataset == 0:
                model_name = experiment["name"] + f"_cv_{i_cross_val+1}"
            else:
                last_model_file_path = model_file_path

            model_name += f"__{'_'.join(dataset_group)}"
            model_file_path = os.path.join(config.models_path, f"{model_name}.pt")

            print(f" > Training model {model_name}")

            # Checking of model already trained exists
            if os.path.exists(model_file_path) and not force_train:
                print(f" > > Found model already trained. Jumping to next dataset...")
                continue

            # Preparing train dataset
            print(f" > Creating Training Dataset for {dataset_group}")
            train_dataset = dataset(
                datasets=dataset_group,
                train_val_test_split=train_val_test_split,
                split="train",
                crop_size=crop_size,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                cross_val_id=i_cross_val,
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
                cross_val_id=i_cross_val,
            )
            val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
            )

            # Initialiazing Tensorboard logging and adding model graph
            print(" > Creating TensorBoard writer and adding graph")
            writer = SummaryWriter(log_dir=os.path.join(config.logs_path, model_name))
            writer.add_graph(
                model,
                torch.zeros((batch_size, 3, *crop_size), dtype=torch.float32).to(config.device),
            )

            # Initialization of keys
            if i_dataset == 0:
                if experiment["architecture_type"] in [
                    "discrete_key_value_bottleneck",
                    "vector_quantizer",
                ]:
                    print("[PHASE-0] Keys Initialization:")

                    # Keys initialization for descrete key-value bottleneck and vector quantizer
                    model.train()
                    model.encoder.train(False)
                    model.decoder.train(False)

                    # Start training
                    with torch.no_grad():
                        for epoch in range(num_epochs_initialization_keys):

                            print(f" > Training epoch {epoch + 1} of {num_epochs_initialization_keys}")

                            # Epoch training
                            pbar = tqdm(train_dataloader)
                            for step, batch in enumerate(pbar):

                                images, labels = batch
                                images = images.to(config.device)

                                # Inference
                                output = model(images)
            else:
                # Loading best model from last training if it is not the first one
                if experiment["architecture_type"] == "discrete_key_value_bottleneck":
                    # Freezing decoder after training with the first set
                    for param in model.decoder.parameters():
                        param.requires_grad = False

                print(f" > Loading model weights from {last_model_file_path}")
                model.load_state_dict(torch.load(last_model_file_path))

            print("[PHASE-1] Training Model:")
            best_val_f1 = 0.0
            patience_iterations = 0
            for epoch in range(num_epochs):

                print(f" > Training epoch {epoch + 1} of {num_epochs}")

                model.train()

                # Configurating model to only train certaing parts of it
                if experiment["architecture_type"] == "discrete_key_value_bottleneck":
                    # All model frozen but values
                    model.train(False)
                    if i_dataset == 0:
                        model.decoder.train(True)
                elif experiment["architecture_type"] == "vector_quantizer":
                    # Freezing Keys and encoder
                    model.train(False)
                    model.decoder.train(True)
                elif experiment["architecture_type"] == "baseline":
                    # Freezing encoder
                    model.train(False)
                    model.decoder.train(True)

                # Epoch training
                running_loss = 0.0
                running_f1 = 0.0
                pbar = tqdm(train_dataloader)
                for step, batch in enumerate(pbar):

                    images, labels = batch
                    images = images.to(config.device)
                    labels = labels.to(config.device)

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
                    avg_loss = running_loss / (step + 1)  # add batch_size
                    running_f1 += f1(
                        output.detach().flatten(),
                        labels.detach().flatten().type(torch.int32),
                    ).item()
                    avg_f1 = running_f1 / (step + 1)  # add batch_size

                    pbar.set_postfix({"loss": avg_loss, "f1": avg_f1})

                # Adding logs for every epoch
                writer.add_scalar("Train Loss", avg_loss, epoch)
                writer.add_scalar("Train F1 Score", avg_f1, epoch)

                print(" > Evaluating validation")
                with torch.no_grad():
                    model.eval()

                    # Epoch validation
                    val_loss, val_f1 = evaluate_dataset_patchwise(
                        model=model,
                        data_loader=val_dataloader,
                        crop_size=crop_size,
                    )

                    # Adding logs for every epoch
                    writer.add_scalar("Validation Loss", avg_loss, epoch)
                    writer.add_scalar("Validation F1 Score", val_f1, epoch)

                    # Evaluting epoch results
                    if val_f1 > best_val_f1:
                        # Saving new best model and initialize variables
                        best_val_f1 = val_f1
                        patience_iterations = 0

                        torch.save(model.state_dict(), model_file_path)

                        print(f" > New best model found with best F1-Score {val_f1} ({val_loss=})  ")
                        print(f" > New best model saved in {model_file_path}")
                    else:
                        # Reducing learning rate and/or stopping the training
                        patience_iterations += 1
                        if (patience_iterations % patience_learning_rate) == 0:
                            learning_rate /= 2.0
                        if patience_iterations >= patience:
                            break

            writer.close()
            print(" > Training model {model_name} ended")


if __name__ == "__main__":
    fire.Fire(train)
