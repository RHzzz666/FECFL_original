from torchvision.datasets import CIFAR10
import numpy as np


class ConcatCIFAR10(CIFAR10):
    def __init__(self, *datasets, transform=None, target_transform=None):
        super(ConcatCIFAR10, self).__init__(root=datasets[0].root, transform=datasets[0].transform,
                                            target_transform=datasets[0].target_transform)

        # Merge data and targets
        self.data = np.concatenate([dataset.data for dataset in datasets])
        self.targets = sum([dataset.targets for dataset in datasets], [])  # Flatten the list of targets
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

