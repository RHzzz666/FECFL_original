from torch.utils.data import Dataset
import os

from PIL import Image
import numpy as np
import torch


class BaseDataset(Dataset):
    """Base dataset iterator"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.data = torch.stack(self.x).numpy()
        self.target = torch.tensor(self.y).numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Subset(Dataset):
    """For data subset with different augmentation for different client.

    Args:
        dataset (Dataset): The whole Dataset
        indices (List[int]): Indices of sub-dataset to achieve from ``dataset``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.data = []
        for idx in indices:
            self.data.append(dataset.data[idx])

        if not isinstance(dataset.targets, np.ndarray):
            dataset.targets = np.array(dataset.targets)

        self.targets = dataset.targets[indices].tolist()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """Get item

        Args:
            index (int): index

        Returns:
            (image, target) where target is index of the target class.
        """
        img, label = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.targets)


class CIFARSubset(Subset):
    """For data subset with different augmentation for different client.

    Args:
        dataset (Dataset): The whole Dataset
        indices (List[int]): Indices of sub-dataset to achieve from ``dataset``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self,
                 dataset,
                 indices,
                 transform=None,
                 target_transform=None,
                 to_image=True):
        self.data = []
        for idx in indices:
            if to_image:
                self.data.append(Image.fromarray(dataset.data[idx]))
        if not isinstance(dataset.targets, np.ndarray):
            dataset.targets = np.array(dataset.targets)
        self.targets = dataset.targets[indices].tolist()
        self.transform = transform
        self.target_transform = target_transform


class FedDataset(object):
    def __init__(self) -> None:
        self.num = None  # the number of dataset indexed from 0 to num-1.
        self.root = None  # the raw dataset.
        self.path = None  # path to save the partitioned datasets.

    def preprocess(self):
        """Define the dataset partition process"""
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

    def get_dataset(self, id, type="train"):
        """Get dataset class

        Args:
            id (int): Client ID for the partial dataset to achieve.
            type (str, optional): Type of dataset, can be chosen from ``["train", "val", "test"]``. Defaults as ``"train"``.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    def get_dataloader(self, id, batch_size, type="train"):
        """Get data loader"""
        raise NotImplementedError()

    def __len__(self):
        return self.num
