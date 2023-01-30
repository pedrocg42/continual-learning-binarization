import glob
import os
from re import S
from typing import Tuple, Union, List

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm

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

    datasets_paths = {
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
        patches_per_image: int = 100,  # max 1000
        cross_val_splits: int = config.cross_val_splits,
        cross_val_id: int = 0,
    ):

        self.datasets = datasets
        self.split = split
        self.train_val_test_split = train_val_test_split
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.patches_per_image = patches_per_image
        self.cross_val_splits = cross_val_splits
        self.cross_val_id = cross_val_id
        self.index = 0

        np.random.seed(config.seed)

        if isinstance(self.datasets, str) and datasets == "all":
            self.datasets = BinDataset.datasets_paths.keys()

        # Extracting input and ground truth images
        self.gr_paths = []
        self.gt_paths = []
        for dataset_name in self.datasets:

            dataset_folder = os.path.join(config.dataset_path, dataset_name)

            gr_images_paths = [
                path
                for path in glob.glob(os.path.join(dataset_folder, "**", "*GR", "*.png"), recursive=True)
                if "aug" not in path
            ]
            gt_images_paths = [
                path
                for path in glob.glob(os.path.join(dataset_folder, "**", "*GT", "*.png"), recursive=True)
                if "aug" not in path
            ]

            # Shufflind paths to create splits
            index = np.arange(len(gr_images_paths))
            np.random.shuffle(index)

            gr_images_paths = np.array(gr_images_paths)[index]
            gt_images_paths = np.array(gt_images_paths)[index]

            # Creating cross-val chunks
            chunk_size = len(gr_images_paths) / config.cross_val_splits
            gr_paths_chunks = [
                gr_images_paths[int(i * chunk_size) : int((i + 1) * chunk_size)] for i in range(config.cross_val_splits)
            ]
            gt_paths_chunks = [
                gt_images_paths[int(i * chunk_size) : int((i + 1) * chunk_size)] for i in range(config.cross_val_splits)
            ]

            test_set_images = gr_paths_chunks.pop(self.cross_val_id)
            val_set_images = gr_paths_chunks.pop(-1)
            train_set_images = np.concatenate(gr_paths_chunks)
            if self.split == "train":
                gr_images_paths = train_set_images
            elif self.split == "val":
                gr_images_paths = val_set_images
            elif self.split == "test":
                gr_images_paths = test_set_images

            test_set_images = gt_paths_chunks.pop(self.cross_val_id)
            val_set_images = gt_paths_chunks.pop(-1)
            train_set_images = np.concatenate(gt_paths_chunks)
            if self.split == "train":
                gt_images_paths = train_set_images
            elif self.split == "val":
                gt_images_paths = val_set_images
            elif self.split == "test":
                gt_images_paths = test_set_images

            if self.split == "train":
                # Loading patches from images paths
                dataset_folder = os.path.join(dataset_folder, "patches")

                if not os.path.isdir(dataset_folder):
                    self.generate_dataset_patches(dataset_name)

                # Loading all patches
                gr_images_names = [os.path.basename(path).split(".")[0] for path in gr_images_paths]
                for i, path in enumerate(
                    glob.glob(os.path.join(dataset_folder, "**", "*GR", "**", "*.png"), recursive=True)
                ):
                    if (
                        int(os.path.basename(path).split(".")[0]) <= self.patches_per_image
                        and os.path.basename(os.path.dirname(path)) in gr_images_names
                    ):
                        self.gr_paths.append(path)

                gt_images_names = [os.path.basename(path).split(".")[0] for path in gt_images_paths]
                for i, path in enumerate(
                    glob.glob(os.path.join(dataset_folder, "**", "*GT", "**", "*.png"), recursive=True)
                ):
                    if (
                        int(os.path.basename(path).split(".")[0]) <= self.patches_per_image
                        and os.path.basename(os.path.dirname(path)) in gt_images_names
                    ):
                        self.gt_paths.append(path)

            elif self.split == "val" or self.split == "test":
                self.gr_paths += list(gr_images_paths)
                self.gt_paths += list(gt_images_paths)

        if self.split == "train":
            # Shufflind paths to create splits
            index = np.arange(len(self.gr_paths))
            np.random.shuffle(index)

            self.gr_paths = np.array(self.gr_paths)[index]
            self.gt_paths = np.array(self.gt_paths)[index]

        elif self.split == "val" or self.split == "test":
            self.gr_paths = np.array(self.gr_paths)
            self.gt_paths = np.array(self.gt_paths)

        self.num_images = len(self.gr_paths)

    def transform(self, image, mask):

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def __len__(self):
        if self.split == "train":
            return self.batch_size * self.steps_per_epoch
        else:
            return self.num_images

    def __getitem__(self, index):

        self.index += 1
        self.index %= self.num_images

        if self.split == "train":
            x, y = self.transform(
                Image.open(self.gr_paths[self.index]).convert("RGB"),
                Image.open(self.gt_paths[self.index]).convert("L"),  # only luminance (one_channel)
            )
            return x, y
        else:
            patches = self.__extract_all_evaluation_patches(np.asarray(Image.open(self.gr_paths[index]).convert("RGB")))

            patches = self.__normalize_image(patches)
            label = TF.to_tensor(Image.open(self.gt_paths[index]).convert("L"))

            return patches, label

    def generate_dataset_patches(self, dataset_name: str, num_patches: int = 1000):

        print(f"Creating patches for dataset {dataset_name}")

        # Creating folder for patches
        dataset_folder = os.path.join(config.dataset_path, dataset_name)
        dataset_patches_folder = os.path.join(dataset_folder, "patches")
        os.makedirs(dataset_patches_folder, exist_ok=True)

        # Finding all the images in the dataset
        gr_paths = [
            path
            for path in glob.glob(os.path.join(dataset_folder, "**", "*GR", "*.png"), recursive=True)
            if "aug" not in path
        ]
        gt_paths = [
            path
            for path in glob.glob(os.path.join(dataset_folder, "**", "*GT", "*.png"), recursive=True)
            if "aug" not in path
        ]

        print(f" > Creating patches for {len(gr_paths)} images found")
        for gr_path, gt_path in tqdm(zip(gr_paths, gt_paths), total=len(gr_paths)):

            gr_image = cv2.imread(gr_path)
            gt_image = cv2.imread(gt_path)

            if os.path.basename(gr_path) != os.path.basename(gt_path):
                raise ValueError("The name of the images should be the same")

            if gr_image.shape[:2] != gt_image.shape[:2]:
                raise ValueError("Images should be the same size")

            height, width = gr_image.shape[:2]

            rand_y = np.random.randint(low=0, high=height - self.crop_size[0], size=num_patches)
            rand_x = np.random.randint(low=0, high=width - self.crop_size[1], size=num_patches)

            # Creating image's patches' folder
            gr_patches_folder = gr_path.replace(dataset_folder, dataset_patches_folder).split(".")[0]
            gt_patches_folder = gt_path.replace(dataset_folder, dataset_patches_folder).split(".")[0]
            os.makedirs(gr_patches_folder, exist_ok=True)
            os.makedirs(gt_patches_folder, exist_ok=True)

            for i, (y, x) in enumerate(zip(rand_y, rand_x)):

                # Extracting patch
                gr_patch = gr_image[y : y + self.crop_size[0], x : x + self.crop_size[1]]
                gt_patch = gt_image[y : y + self.crop_size[0], x : x + self.crop_size[1]]

                # Creating patches paths and saving them
                gr_patch_path = os.path.join(gr_patches_folder, f"{(i+1):04d}.png")
                gt_patch_path = os.path.join(gt_patches_folder, f"{(i+1):04d}.png")
                cv2.imwrite(gr_patch_path, gr_patch)
                cv2.imwrite(gt_patch_path, gt_patch)

    def __extract_all_evaluation_patches(self, image: np.ndarray):
        gr_patches = []

        height, width = image.shape[:2]

        pt_image = np.transpose(image, axes=[2, 0, 1])

        # Calculating necessary patches
        num_patches_height = np.ceil(height / self.crop_size[0]).astype(int)
        num_patches_width = np.ceil(width / self.crop_size[1]).astype(int)

        for i in range(num_patches_height):
            for j in range(num_patches_width):

                top = i * self.crop_size[0]
                left = j * self.crop_size[1]

                # Avoiding the patch to be outside the image
                if top + self.crop_size[0] >= height:
                    top = height - self.crop_size[0]
                if left + self.crop_size[1] >= width:
                    left = width - self.crop_size[1]

                gr_patches.append(pt_image[:, top : top + self.crop_size[0], left : left + self.crop_size[1]])

        gr_patches = torch.from_numpy(np.array(gr_patches, dtype=np.float32))

        return gr_patches

    def __normalize_image(self, image):
        return (image / 127.5) - 1.0
