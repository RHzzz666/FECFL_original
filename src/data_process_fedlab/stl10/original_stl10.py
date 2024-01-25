import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms import functional

from ..basic_dataset import FedDataset, BaseDataset
from ..functional import noniid_slicing, random_slicing, plot_label_distribution, \
    noniid_slicing_with_dirichlet, noniid_slicing_label_skew, pathological_slicing, plot_partitions, \
    pathological_slicing_2
from .full_loader_stl10 import ConcatSTL10


class OriginalSTL10(FedDataset):
    """Label change for STL10 and patrition them.

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
        # "./datasets/labels_changed_stl10/"
        if os.path.exists(save_dir) is not True:
            os.makedirs(save_dir, exist_ok=True)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preprocess(self, labels_mapping=None):
        print("########### Preprocessing original STL10 ###########")
        stl10_train = torchvision.datasets.STL10(root=self.root, split='train', download=True)
        stl10_test = torchvision.datasets.STL10(root=self.root, split='test', download=True)
        stl10_full = ConcatSTL10(stl10_train, stl10_test)
        if self.partition_method > "noniid-#label0" and self.partition_method <= "noniid-#label9":
            num_label = eval(self.partition_method[13:])
            partitions = noniid_slicing_label_skew(stl10_full, self.num, 10, num_label)
        elif self.partition_method == "noniid-#dirichlet":
            partitions = noniid_slicing_with_dirichlet(stl10_full, self.num, 10, self.alpha)
        elif self.partition_method == "pathological#2":
            groups = [[0, 2], [1, 3], [4, 5], [6, 7], [8, 9]]
            partitions = pathological_slicing_2(stl10_full, self.num, groups)
        elif self.partition_method == "homo":
            partitions = random_slicing(stl10_full, self.num)
        else:
            raise ValueError("partition_method must be one of ['noniid', 'class_group', 'homo', 'allsame']")

        plot_partitions(partitions, stl10_full.targets)

        original_data = []
        original_labels = []
        for x, y in stl10_full:
            x = np.transpose(x, (1, 2, 0))
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
