# Taken from Bhoumik
import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from pathlib import Path

# Parameters in the code:
# csv_file: image path from protocol
# image_path: directory of the dataset
# transform: which transformation to use in the image_path
# method: softmax, softmax_garbage, entropic. Refers to the loss function.


class Imagenet_dataset(Dataset):
    def __init__(self, csv_file, images_path, transform=None):
        """
        Args:
            csv_file (Path): Path to the csv file with image paths.
            images_path (Path): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            total_labels: Number of labels present in the dataset, includes unknowns.
        """
        self.dataset = pd.read_csv(csv_file, header=None)
        self.images_path = Path(images_path)
        self.transform = transform
        self.label_cnt = len(self.dataset[1].unique())
        self.unique_classes = np.sort(self.dataset[1].unique())

    # return size of dataset
    def __len__(self):
        return len(self.dataset)

    # return sample from the dataset given an index
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            print("index is tensor")
        if not isinstance(index, int):
            print("Index is an iterable")
        # path to image within a class from csv file
        # todo: check if there is a better way to access the locations.
        jpeg_path = self.dataset.iloc[index, 0]
        # corresponding integer label for image
        label = self.dataset.iloc[index, 1]
        # use RGB mode to represent and store the image
        x = Image.open(self.images_path/jpeg_path).convert('RGB')
        # apply image transformation
        if self.transform is not None:
            x = self.transform(x)
        # convert integer label to tensor
        t = torch.as_tensor(int(label), dtype=torch.int64)
        # return image, label
        return x, t

    def has_unknowns(self):
        return -1 in self.unique_classes
