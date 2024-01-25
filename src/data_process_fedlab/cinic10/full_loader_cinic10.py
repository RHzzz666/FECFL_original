from torchvision.datasets import ImageFolder


class ConcatCINIC10(ImageFolder):
    def __init__(self, *datasets, transform=None, target_transform=None):
        super().__init__(root=datasets[0].root, transform=transform,
                         target_transform=target_transform)

        # Merge image paths and targets from all datasets
        self.imgs = []
        self.targets = []
        for dataset in datasets:
            self.imgs.extend(dataset.imgs)
            self.targets.extend([target for _, target in dataset.imgs])

    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
