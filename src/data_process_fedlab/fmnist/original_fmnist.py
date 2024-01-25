
import logging
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from ..basic_dataset import FedDataset, BaseDataset
from ..functional import noniid_slicing, random_slicing, plot_label_distribution, \
    noniid_slicing_with_dirichlet, noniid_slicing_label_skew, pathological_slicing, plot_partitions
from .full_loader_fmnist import ConcatFMNIST


class OriginalFMNIST(FedDataset):
    def __init__(self, root, path, num, partition_method="noniid", alpha=0.5) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num = num
        self.partition_method = partition_method
        self.alpha = alpha

    def preprocess(self, thetas=[0, 90, 180, 270], download=True):
        print("########### Preprocessing rotated FMNIST ###########")
        self.download = download
        # "./datasets/rotated_fmnist/"
        if os.path.exists(self.path) is not True:
            os.makedirs(self.path, exist_ok=True)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "test"))

        fmnist_train = torchvision.datasets.FashionMNIST(self.root, train=True, download=self.download)
        fmnist_test = torchvision.datasets.FashionMNIST(self.root, train=False, download=self.download)
        fmnist_full = ConcatFMNIST(fmnist_train, fmnist_test)

        if self.partition_method > "noniid-#label0" and self.partition_method <= "noniid-#label9":
            num_label = eval(self.partition_method[13:])
            partitions = noniid_slicing_label_skew(fmnist_full, self.num, 10, num_label)
        elif self.partition_method == "noniid-labeldir":
            partitions = noniid_slicing_with_dirichlet(fmnist_full, self.num, 10, self.alpha)
        elif self.partition_method == "pathological":
            partitions = pathological_slicing(fmnist_full, self.num, 10, 2)
        elif self.partition_method == "homo":
            partitions = random_slicing(fmnist_full, self.num)
        else:
            raise ValueError("partition_method must be one of ['noniid', 'class_group', 'homo', 'allsame']")

        plot_partitions(partitions, fmnist_full.targets)

        to_tensor = transforms.ToTensor()
        original_data = []
        original_labels = []
        for x, y in fmnist_full:
            x = to_tensor(x)
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
        ds_size = len(dataset)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        return data_loader, ds_size
