from PIL import Image
import os

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
from torchvision import transforms
from torchvision.transforms import functional

from ..basic_dataset import FedDataset, BaseDataset
from ..functional import noniid_slicing, random_slicing, plot_label_distribution, \
    noniid_slicing_with_dirichlet, noniid_slicing_label_skew, pathological_slicing, plot_partitions, \
    pathological_slicing_2


class TinyImageNet(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Root directory of the dataset.
            split (string, optional): The dataset split, either 'train' or 'val'.
            transform (callable, optional): An optional transform to be applied on a sample. You can pass any torchvision.transforms operation here.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        # Load paths and labels of the dataset
        if split == 'train':
            for class_id, class_name in enumerate(sorted(os.listdir(os.path.join(root_dir, 'train')))):
                for image_name in os.listdir(os.path.join(root_dir, 'train', class_name, 'images')):
                    self.images.append(os.path.join(root_dir, 'train', class_name, 'images', image_name))
                    self.labels.append(class_id)
        elif split == 'val':
            # The handling for the validation set might need adjustments based on your dataset's file structure
            with open(os.path.join(root_dir, 'val', 'val_annotations.txt'), 'r') as f:
                for line in f.readlines():
                    image_name, class_name = line.split('\t')[:2]
                    image_path = os.path.join(root_dir, 'val', 'images', image_name)
                    class_id = sorted(os.listdir(os.path.join(root_dir, 'train'))).index(class_name)
                    self.images.append(image_path)
                    self.labels.append(class_id)

    def __len__(self):
        """Return the number of data points in the dataset"""
        return len(self.images)

    def __getitem__(self, idx):
        """Return a sample from the dataset given an index idx"""
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class OriginalTINY(FedDataset):
    """Label change for TINY and patrition them.

        Args:
            root (str): Path to download raw dataset.
            path (str): Path to save partitioned subdataset.
            num_clients (int): Number of clients.
    """

    def __init__(self, root, save_dir, num_clients, partition_method="noniid", alpha=0.1):
        self.root = os.path.expanduser(root)
        self.path = save_dir
        self.num = num_clients
        self.partition_method = partition_method
        self.alpha = alpha
        self.id_dict = {}
        if os.path.exists(save_dir) is not True:
            os.makedirs(save_dir, exist_ok=True)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def preprocess(self, labels_mapping=None):
        print("########### Preprocessing original TINY ###########")
        for i, line in enumerate(open(self.root + '/wnids.txt', 'r')):
            self.id_dict[line.replace('\n', '')] = i

        # Load the TINY dataset
        tiny_train = TinyImageNet(self.root, 'train')
        tiny_test = TinyImageNet(self.root, 'val')

        # Concatenate the training and test datasets
        tiny_full = ConcatDataset([tiny_train, tiny_test])

        train_targets = [label for _, label in [tiny_train[i] for i in range(len(tiny_train))]]
        test_targets = [label for _, label in [tiny_test[i] for i in range(len(tiny_test))]]
        tiny_full.targets = train_targets + test_targets

        if self.partition_method > "noniid-#label0" and self.partition_method <= "noniid-#label9":
            num_label = eval(self.partition_method[13:])
            partitions = noniid_slicing_label_skew(tiny_full, self.num, 10, num_label)
        elif self.partition_method == "noniid-#dirichlet":
            partitions = noniid_slicing_with_dirichlet(tiny_full, self.num, 10, self.alpha)
        elif self.partition_method == "pathological":
            partitions = pathological_slicing(tiny_full, self.num, 200, 50)
        elif self.partition_method == "homo":
            partitions = random_slicing(tiny_full, self.num)
        else:
            raise ValueError("partition_method must be one of ['noniid', 'class_group', 'homo', 'allsame']")

        # plot_label_distribution(self.num, 10, partitions, cifar10_full.targets)
        plot_partitions(partitions, tiny_full.targets)

        original_data = []
        original_labels = []
        for x, y in tiny_full:
            x = self.transform(x)
            original_data.append(x)
            original_labels.append(y)

        for client_idx in range(self.num):
            partition = partitions[client_idx]
            data_num_client = len(partition)
            # logging.info("data_num_client: %s" % data_num_client)
            train_idx = partition[:int(data_num_client * 0.8)]
            test_idx = partition[int(data_num_client * 0.8):]

            # logging.info("client_%s train data : %s" % (client_idx, len(train_idx)))
            # logging.info("client_%s test data : %s" % (client_idx, len(test_idx)))

            data_train = [original_data[i] for i in train_idx]
            label_train = [original_labels[i] for i in train_idx]
            dataset_train = BaseDataset(data_train, label_train)
            # logging.info("len of dataset_train: {}".format(len(dataset_train)))
            torch.save(dataset_train, os.path.join(self.path, "train", "data{}.pkl".format(client_idx)))

            data_test = [original_data[i] for i in test_idx]
            label_test = [original_labels[i] for i in test_idx]
            dataset_test = BaseDataset(data_test, label_test)
            # logging.info("len of dataset_test: {}".format(len(dataset_test)))
            torch.save(dataset_test, os.path.join(self.path, "test", "data{}.pkl".format(client_idx)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
