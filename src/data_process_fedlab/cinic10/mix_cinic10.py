
import os
import torch

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import functional

from ..basic_dataset import FedDataset, BaseDataset
from ..functional import noniid_slicing, random_slicing, plot_label_distribution, plot_partitions, pathological_slicing, pathological_slicing_3

from .full_loader_cinic10 import ConcatCINIC10


class MixCINIC10(FedDataset):
    """MixCIFAR10 and patrition them.

        Args:
            root (str): Path to download raw dataset.
            path (str): Path to save partitioned subdataset.
            num_clients (int): Number of clients.
        """

    def __init__(self, root, save_dir, num_clients):
        self.root = os.path.expanduser(root)
        self.path = save_dir
        self.num = num_clients
        # "./datasets/rotated_cifar10/"
        if os.path.exists(save_dir) is not True:
            os.makedirs(save_dir, exist_ok=True)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preprocess(self, thetas=[0, 90, 180, 270, 0, 90, 180, 270]):
        """_summary_

        Args:
            shards (_type_): _description_
            thetas (list, optional): _description_. Defaults to [0, 180].
        """
        # Load the CIFAR-10 dataset
        print("########### Preprocessing rotated CIFAR10 ###########")
        cinic10_train = torchvision.datasets.ImageFolder(root=os.path.join(self.root, "train"))
        cinic10_test = torchvision.datasets.ImageFolder(root=os.path.join(self.root, "test"))
        # Concatenate the training and test datasets
        cinic10_full = ConcatCINIC10(cinic10_train, cinic10_test)
        partitions = pathological_slicing(cinic10_full, self.num, 10, 5)

        # plot_label_distribution(self.num, 10, partitions, cifar10_full.targets)
        plot_partitions(partitions, cinic10_full.targets)

        for group_idx, theta in enumerate(thetas):
            rotated_data = []
            labels = []
            for x, y in cinic10_full:
                x = self.transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
                labels.append(y)

            for client_idx in range(self.num):
                # logging.info("client_idx: %s" % client_idx)
                # logging.info("client_idx // (self.num / len(thetas)): %s "
                #              % (client_idx // (self.num / len(thetas))))
                # logging.info("group_idx: %s" % group_idx)
                if client_idx // (self.num / len(thetas)) == group_idx:
                    partition = partitions[client_idx]
                    data_num_client = len(partition)
                    # logging.info("data_num_client: %s" % data_num_client)
                    train_idx = partition[:int(data_num_client * 0.8)]
                    test_idx = partition[int(data_num_client * 0.8):]

                    # logging.info("client_%s train data : %s" % (client_idx, len(train_idx)))
                    # logging.info("client_%s test data : %s" % (client_idx, len(test_idx)))

                    data_train = [rotated_data[i] for i in train_idx]
                    label_train = [labels[i] for i in train_idx]
                    dataset_train = BaseDataset(data_train, label_train)
                    # logging.info("len of dataset_train: {}".format(len(dataset_train)))
                    torch.save(dataset_train, os.path.join(self.path, "train", "data{}.pkl".format(client_idx)))

                    data_test = [rotated_data[i] for i in test_idx]
                    label_test = [labels[i] for i in test_idx]
                    dataset_test = BaseDataset(data_test, label_test)
                    # logging.info("len of dataset_test: {}".format(len(dataset_test)))
                    torch.save(dataset_test, os.path.join(self.path, "test", "data{}.pkl".format(client_idx)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        ds_size = len(dataset)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        return data_loader, ds_size
