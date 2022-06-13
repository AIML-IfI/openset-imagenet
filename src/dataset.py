# Code based on: Bhoumik, A. (2021). Open-set Classification on ImageNet. Master’s thesis, University of Zurich.
import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from pathlib import Path


class ImagenetDataset(Dataset):
    """ Imagenet Dataset. """

    def __init__(self, csv_file, images_path, transform=None):
        """ Constructs an Imagenet Dataset from a CSV file. The file should list the path to the images and
        the corresponding label. For example: val/n02100583/ILSVRC2012_val_00013430.JPEG,   0
        Args:
            csv_file (Path): Path to the csv file with image paths and labels.
            images_path (Path): Home directory of the Imagenet dataset.
            transform (callable, optional): Transformations to apply while loading the images.
        """
        self.dataset = pd.read_csv(csv_file, header=None)
        self.images_path = Path(images_path)
        self.transform = transform
        self.label_cnt = len(self.dataset[1].unique())
        self.unique_classes = np.sort(self.dataset[1].unique())

    def __len__(self):
        """ Returns the length of the dataset. """
        return len(self.dataset)

    def __getitem__(self, index):
        """ Returns a tuple (image, label) of the dataset at the given index. If available, it applies
        the defined transformations to the image. Images are converted to RGB format.
        Args:
            index (Int): Image index
        Returns: Tuple of tensors (image, label)
        """
        if torch.is_tensor(index):
            # print("index is tensor")
            index = index.tolist()
        jpeg_path = self.dataset.iloc[index, 0]
        label = self.dataset.iloc[index, 1]
        x = Image.open(self.images_path/jpeg_path).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        # convert integer label to tensor
        t = torch.as_tensor(int(label), dtype=torch.int64)
        return x, t

    def has_unknowns(self):
        """ Returns true if the dataset contains known-unknown samples."""
        return -1 in self.unique_classes

    def replace_unknown_label(self):
        """Replaces unknown label (-1) to the biggest label + 1. This is required if the loss function
        is softmax with garbage class. Updates the array of unique labels."""
        biggest_label = self.unique_classes[-1]
        self.dataset[1].replace(-1, biggest_label+1, inplace=True)
        self.unique_classes[self.unique_classes == -1] = biggest_label+1
        self.unique_classes.sort()

    def calculate_class_weights(self):
        counts = self.dataset.groupby(1).count().to_numpy()
        class_weights = (len(self.dataset)/(counts * self.label_cnt))
        return torch.from_numpy(class_weights).float()
