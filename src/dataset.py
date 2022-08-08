""" Code based on: Bhoumik, A. (2021). Open-set Classification on ImageNet."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset


class ImagenetDataset(Dataset):
    """ Imagenet Dataset. """

    def __init__(self, csv_file, imagenet_path, transformation=None):
        """ Constructs an Imagenet Dataset from a CSV file. The file should list the path to the
        images and the corresponding label. For example:
        val/n02100583/ILSVRC2012_val_00013430.JPEG,   0

        Args:
            csv_file(Path): Path to the csv file with image paths and labels.
            imagenet_path(Path): Home directory of the Imagenet dataset.
            transformation(torchvision.transforms): Transformations to apply to the images.
        """
        self.dataset = pd.read_csv(csv_file, header=None)
        self.imagenet_path = Path(imagenet_path)
        self.transformation = transformation
        self.label_count = len(self.dataset[1].unique())
        self.unique_classes = np.sort(self.dataset[1].unique())

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.dataset)

    def __getitem__(self, index):
        """ Returns a tuple (image, label) of the dataset at the given index. If available, it
        applies the defined transformations to the image. Images are converted to RGB format.

        Args:
            index(int): Image index

        Returns:
            image, label: (image tensor, label tensor)
        """
        if torch.is_tensor(index):
            index = index.tolist()

        jpeg_path, label = self.dataset.iloc[index]
        image = Image.open(self.imagenet_path / jpeg_path).convert("RGB")

        if self.transformation is not None:
            image = self.transformation(image)

        # convert int label to tensor
        label = torch.as_tensor(int(label), dtype=torch.int64)
        return image, label

    def has_negatives(self):
        """ Returns true if the dataset contains negative samples."""
        return -1 in self.unique_classes

    def replace_negative_label(self):
        """ Replaces negative label (-1) to biggest_label + 1. This is required if the loss function
        is BGsoftmax. Updates the array of unique labels.
        """
        biggest_label = self.unique_classes[-1]
        self.dataset[1].replace(-1, biggest_label + 1, inplace=True)
        self.unique_classes[self.unique_classes == -1] = biggest_label + 1
        self.unique_classes.sort()

    def calculate_class_weights(self):
        """ Calculate the class weights based on sample counts.

        Returns:
            class_weights: Tensor with weight for every class.
        """
        # TODO: Should it be part of dataset class?
        counts = self.dataset.groupby(1).count().to_numpy()
        class_weights = (len(self.dataset) / (counts * self.label_count))
        return torch.from_numpy(class_weights).float()
