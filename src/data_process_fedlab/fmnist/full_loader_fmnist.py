import numpy as np
import torch
from torchvision.datasets import FashionMNIST


class ConcatFMNIST(FashionMNIST):
    def __init__(self, *datasets, transform=None, target_transform=None):
        super(ConcatFMNIST, self).__init__(root=datasets[0].root, train=datasets[0].train,
                                          transform=datasets[0].transform,
                                          target_transform=datasets[0].target_transform,
                                          download=datasets[0].download)

        # Merge data and targets
        self.data = np.concatenate([dataset.data for dataset in datasets])
        # Assuming `datasets` is a list of dataset objects
        self.targets = []
        for dataset in datasets:
            # Convert targets to a list if they are tensors
            targets_list = dataset.targets.tolist() if isinstance(dataset.targets, torch.Tensor) else dataset.targets
            self.targets += targets_list  # Concatenate the lists

        if transform is not None:
            self.transform = transform
        if target_transform is not None:
            self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)
