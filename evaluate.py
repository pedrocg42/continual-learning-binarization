import csv
import os
from typing import List, Tuple, Union

import fire
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from my_utils.evaluate import evaluate_dataset_patchwise
from my_utils.parse_experiment import parse_experiment


@parse_experiment
def evaluate(
    architecture,
    dataset: Dataset,
    datasets: Union[List[str], str],
    train_val_test_split: List[float],
    crop_size: Tuple[int],
    batch_size: int,
    steps_per_epoch: int,
    all_datasets: List[str] = config.ALL_DATASETS,
    **experiment,
):

    print(f"Training experiment: {experiment['name']}")

    # Building the model
    print("Building architecture")
    model = architecture(**experiment)
    model.to(config.device)
    print(model)
    print(
        f"Encoder total parameters: {sum(param.numel() for param in model.parameters())}"
    )

    # Laoding different models from an experiment
    for i_cross_val in range(config.cross_val_splits):
        model_results = {}
        for i, dataset_group in enumerate(datasets):

            if i == 0:
                model_name = experiment["name"] + f"_cv_{i_cross_val+1}"

            model_name = model_name + f"__{'_'.join(dataset_group)}"
            model_results[model_name] = {}

            model_file_path = os.path.join(config.models_path, f"{model_name}.pt")
            print(f" > Loading model from {model_file_path}")
            model.load_state_dict(torch.load(model_file_path))

            # Extracting results for all the datasets

            for dataset_name in all_datasets:

                model_results[model_name][dataset_name[0]] = {}

                # Preparing testing dataset
                print(f" > Creating Testing Dataset {dataset_name}")
                test_dataset = dataset(
                    datasets=dataset_name,
                    train_val_test_split=train_val_test_split,
                    split="test",
                    crop_size=crop_size,
                    batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch,
                    cross_val_id=i_cross_val,
                )

                model.eval()

                mse, f1 = evaluate_dataset_patchwise(
                    model=model,
                    dataset=test_dataset,
                    crop_size=crop_size,
                )

                model_results[model_name][dataset_name[0]]["mse"] = mse
                model_results[model_name][dataset_name[0]]["f1"] = f1

        # Extracting metrics
        print(f" > Saving metrics...")
        csv_path = "results.csv"
        csv_existed = os.path.exists(csv_path)

        f = open(csv_path, "a")
        csv_writer = csv.writer(f, delimiter="\t")

        if not csv_existed:
            # Creating header
            header = ["Experiment", "F1-Score", "MSE"]
            header_f1 = []
            header_mse = []
            for dataset_name in all_datasets:
                header_f1.append(f"F1 {dataset_name[0]}")
                header_mse.append(f"MSE {dataset_name[0]}")
            header += header_f1 + header_mse

            # Writing header
            csv_writer.writerow(header)

        # Writing results for every model and for each dataset
        for model_name in model_results.keys():
            row_results_f1 = []
            row_results_mse = []
            for dataset_name in model_results[model_name].keys():
                row_results_f1.append(model_results[model_name][dataset_name]["f1"])
                row_results_mse.append(model_results[model_name][dataset_name]["mse"])

            row_results = (
                [model_name, np.mean(row_results_f1), np.mean(row_results_mse)]
                + row_results_f1
                + row_results_mse
            )
            csv_writer.writerow(row_results)

        print(f" > Results saved in {csv_path}")

        # Close file
        f.close()

    print("End")


if __name__ == "__main__":
    fire.Fire(evaluate)
