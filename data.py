import glob
import os
from re import S
from typing import Tuple, Union, List

import imagesize
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

import config

## DATASETS
#   TEXT:
#       - PHI: a collection of scanned images of Persian manuscripts from the
#           Persian Heritage Image Binarization Competition.
#       - Dibco: set of images of handwritten Latin text documentsfrom the
#           Document Image Binarization Contest, annually held from 2009. We
#           combined the collectionsfrom 2009 to 2016 as a single corpus for
#           our experiments.
#       - Palm: ICFHR 2016 Binarization of Palm Leaf Manuscript Images challenge.
#   MUSIC:
#       - Salzinnes: high-resolution images of scanned documents that contains
#           lyrics and music scores in neumatic notation.
#       - Einsieldeln: high-resolution images of scanned documents that contains
#           lyrics and music scores in neumatic notation.


class BinDataset(Dataset):

    databases_paths = {
        "Dibco": {
            "train": {
                "GR": [
                    "2009/handwritten_GR",
                    "2009/printed_GR",
                    "2010/handwritten_GR",
                    "2011/handwritten_GR",
                    "2011/printed_GR",
                    "2012/handwritten_GR",
                    "2013/handwritten_GR",
                    "2013/printed_GR",
                    "2014/handwritten_GR",
                    "2016/handwritten_GR",
                ],
                "GT": [
                    "2009/handwritten_GT",
                    "2009/printed_GR",
                    "2010/handwritten_GT",
                    "2011/handwritten_GT",
                    "2011/printed_GT",
                    "2012/handwritten_GT",
                    "2013/handwritten_GT",
                    "2013/printed_GT",
                    "2014/handwritten_GT",
                    "2016/handwritten_GT",
                ],
            },
            "test": {
                "GR": [],
                "GT": [],
            },
        },
        "Einsieldeln": {
            "train": {
                "GR": ["ein_GR"],
                "GT": ["ein_GT"],
            },
            "test": {
                "GR": ["ein_GR"],
                "GT": ["ein_GT"],
            },
        },
        "Palm": {
            "train": {
                "GR": [
                    # "gt1_GR",
                    "gt2_GR",
                ],
                "GT": [
                    # "gt1_GT",
                    "gt2_GT",
                ],
            },
            "test": {
                "GR": [
                    # "gt1_GR",
                    "gt2_GR",
                ],
                "GT": [
                    # "gt1_GT",
                    "gt2_GT",
                ],
            },
        },
        "PHI": {
            "train": {
                "GR": ["phi_GR"],
                "GT": ["phi_GT"],
            },
            "test": {
                "GR": ["phi_GR"],
                "GT": ["phi_GT"],
            },
        },
        "Salzinnes": {
            "train": {
                "GR": ["sal_GR"],
                "GT": ["sal_GT"],
            },
            "test": {
                "GR": ["sal_GR"],
                "GT": ["sal_GT"],
            },
        },
    }

    def __init__(
        self,
        datasets: Union[List[str], str] = "all",
        split: str = "train",
        train_val_test_split=[0.6, 0.2, 0.2],
        crop_size: Tuple[int] = (256, 256),
        batch_size: int = 64,
        steps_per_epoch: int = 100,
        seed: int = 42,
    ):

        self.datasets = datasets
        self.split = split
        self.train_val_test_split = train_val_test_split
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        np.random.seed(seed)

        if isinstance(self.datasets, str) and datasets == "all":
            self.datasets = BinDataset.databases_paths.keys()

        # Extractin input and ground truth images for the split
        self.images_paths = []
        self.gt_paths = []
        for database_name in self.datasets:
            # loading all sets then making our own split
            temp_images_paths = []
            temp_gt_paths = []
            for temp_split in ["train", "test"]:
                database_dict = BinDataset.databases_paths[database_name][temp_split]
                for folder_path in database_dict["GR"]:

                    temp_images_paths += glob.glob(
                        os.path.join(config.dataset_path, database_name, temp_split, folder_path, "*.png")
                    )
                for folder_path in database_dict["GT"]:
                    temp_gt_paths += glob.glob(
                        os.path.join(config.dataset_path, database_name, temp_split, folder_path, "*.png")
                    )

            # Shufflind paths to create splits
            index = np.arange(len(temp_images_paths))
            np.random.shuffle(index)

            temp_images_paths = np.array(temp_images_paths)[index]
            temp_gt_paths = np.array(temp_gt_paths)[index]

            # Creating split for each dataset
            idx1 = int(len(temp_images_paths) * self.train_val_test_split[0])
            idx2 = idx1 + int(len(temp_images_paths) * self.train_val_test_split[1])
            if self.split == "train":
                temp_images_paths = temp_images_paths[:idx1]
                temp_gt_paths = temp_gt_paths[:idx1]
            elif self.split == "val":
                temp_images_paths = temp_images_paths[idx1:idx2]
                temp_gt_paths = temp_gt_paths[idx1:idx2]
            elif self.split == "test":
                temp_images_paths = temp_images_paths[idx2:]
                temp_gt_paths = temp_gt_paths[idx2:]

            self.images_paths.append(temp_images_paths)
            self.gt_paths.append(temp_gt_paths)

        self.images_paths = np.concatenate(self.images_paths)
        self.gt_paths = np.concatenate(self.gt_paths)

        self.num_images = len(self.images_paths)

        # For evaluation we need to evaluate all the image
        self.images_sizes = []
        if self.split == "val" or self.split == "test":
            self.patches = []
            temp_images_list = []
            temp_gt_list = []
            for image_path, gt_path in zip(self.images_paths, self.gt_paths):
                self.images_sizes.append(imagesize.get(image_path))
                width, height = self.images_sizes[-1]

                # Calculating necessary patches
                num_patches_height = np.ceil(height / crop_size[0]).astype(int)
                num_patches_width = np.ceil(width / crop_size[1]).astype(int)

                for i in range(num_patches_height):
                    for j in range(num_patches_width):

                        top = i * crop_size[0]
                        left = j * crop_size[1]

                        # Avoiding the patch to be outside the image
                        if top + self.crop_size[0] >= height:
                            top = height - self.crop_size[0]
                        if left + self.crop_size[1] >= width:
                            left = width - self.crop_size[1]

                        temp_images_list.append(image_path)
                        temp_gt_list.append(gt_path)
                        self.patches.append((top, left, *self.crop_size))

            self.images_paths = temp_images_list
            self.gt_paths = temp_gt_list

        self.num_patches = len(self.images_paths)

    def transform(self, image, mask):

        # Random crop
        top, left, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, top, left, h, w)
        mask = TF.crop(mask, top, left, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def transform_eval(self, image: Image.Image, mask: Image.Image, patch: Tuple):

        # Crop image and ground truth
        image = TF.crop(image, *patch)
        mask = TF.crop(mask, *patch)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def __len__(self):
        if self.split == "train":
            return self.batch_size * self.steps_per_epoch
        else:
            return self.num_patches

    def __getitem__(self, index):

        if self.split == "train":
            circular_index = index % self.num_images
            x, y = self.transform(
                Image.open(self.images_paths[circular_index]).convert("RGB"),
                Image.open(self.gt_paths[circular_index]).convert("L"),  # only luminance (one_channel)
            )
        else:
            x, y = self.transform_eval(
                Image.open(self.images_paths[index]).convert("RGB"),
                Image.open(self.gt_paths[index]).convert("L"),  # only luminance (one_channel)
                self.patches[index],
            )

        return x, y
