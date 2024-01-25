from torchvision.datasets import STL10
import numpy as np


class ConcatSTL10(STL10):
    def __init__(self, *datasets, transform=None, target_transform=None):
        super(ConcatSTL10, self).__init__(root=datasets[0].root, transform=datasets[0].transform,
                                          target_transform=datasets[0].target_transform)
        self.data = np.concatenate([dataset.data for dataset in datasets])
        self.targets = np.concatenate([dataset.labels for dataset in datasets])
        if transform is not None:
            self.transform = transform
        if target_transform is not None:
            self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.targets)

