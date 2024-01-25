import copy
import gc

import numpy as np
from math import ceil
import random
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms


# class CustomConcatDataset(ConcatDataset):
#     def __init__(self, datasets):
#         super(CustomConcatDataset, self).__init__(datasets)
#         self.target = []
#
#         for dataset in datasets:
#             if hasattr(dataset, 'target'):
#                 self.target.extend(dataset.target)
#             else:
#                 for _, t in dataset:
#                     self.target.append(t)
#
#
class CustomSubset_incre(Subset):
    def __init__(self, dataset, indices):
        super(CustomSubset_incre, self).__init__(dataset, indices)
        self.target = np.array([], dtype=int)

        self.data = np.array([])

        if hasattr(dataset, 'target'):
            for i in indices:
                self.target = np.append(self.target, dataset.target[i])
        if hasattr(dataset, 'data'):
            for i in indices:
                element = np.array(dataset.data[i])
                if self.data.size == 0:
                    self.data = np.empty((0,) + element.shape)
                self.data = np.append(self.data, [element], axis=0)
        else:
            for i in indices:
                d, t = dataset[i]
                self.target = np.append(self.target, t)
                element = np.array(d)
                if self.data.size == 0:
                    self.data = np.empty((0,) + element.shape)
                self.data = np.append(self.data, [element], axis=0)


class CustomSubset:
    def __init__(self, dataset, indices):
        self.data = [dataset[i][0] for i in indices]
        self.target = [dataset[i][1] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class CustomConcatDataset:
    def __init__(self, datasets):
        self.data = []
        self.target = []

        for dataset in datasets:
            self.data.extend([item[0] for item in dataset])
            self.target.extend([item[1] for item in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]




class RotatedDataset(Dataset):
    def __init__(self, original_dataset, rotate_angle=0):
        self.original_dataset = original_dataset
        self.rotate_angle = rotate_angle
        self.target = []

        if hasattr(original_dataset, 'target'):
            self.target = original_dataset.target
        else:
            for _, t in original_dataset:
                self.target.append(t)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        if self.rotate_angle != 0:
            x = TF.rotate(x, self.rotate_angle)
        return x, y


class FlippedDataset(Dataset):
    def __init__(self, original_dataset, horizontal_flip=False, vertical_flip=False):
        self.original_dataset = original_dataset
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.target = []

        if hasattr(original_dataset, 'targets'):  # 注意这里是 'targets' 而不是 'target'
            self.target = original_dataset.targets
        else:
            for _, t in original_dataset:
                self.target.append(t)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        if self.horizontal_flip:
            x = TF.hflip(x)
        if self.vertical_flip:
            x = TF.vflip(x)
        return x, y


"""swap: change ds and dl"""


def swap_data_all(clients, swap_p):
    # Swap the data of warm clients with probability swap_p
    clients_size = len(clients)
    # Randomly swap two clients' dataset
    if swap_p > 0 and swap_p < 1:
        # Shuffle the client index
        shuffle_idx = np.random.permutation(clients_size)
        swap_flag = np.random.choice([0, 1], int(clients_size / 2), p=[1 - swap_p, swap_p])  # Half size of clients_size
        for idx in np.nonzero(swap_flag)[0]:
            # Swap clients' data with index are idx and -(idx+1)
            cidx1, cidx2 = shuffle_idx[idx], shuffle_idx[-(idx + 1)]
            print("client {} and client {} swap all data".format(cidx1, cidx2))
            c1, c2 = clients[cidx1], clients[cidx2]
            if c1.ds_id is None:
                c1.set_ds_id(cidx1)
            if c2.ds_id is None:
                c2.set_ds_id(cidx2)
            c1.ds_train, c2.ds_train = c2.ds_train, c1.ds_train
            c1.ds_test, c2.ds_test = c2.ds_test, c1.ds_test
            c1.refresh_dl()
            c2.refresh_dl()
            c1_id, c2_id = c1.ds_id, c2.ds_id
            c1.set_ds_id(c2_id)
            c2.set_ds_id(c1_id)


def swap_data_part(clients, swap_p):
    # Swap the data of warm clients with probability swap_p
    clients_size = len(clients)
    # Randomly swap two clients' dataset
    if swap_p > 0 and swap_p < 1:
        # Shuffle the client index
        shuffle_idx = np.random.permutation(clients_size)
        swap_flag = np.random.choice([0, 1], int(clients_size / 2), p=[1 - swap_p, swap_p])  # Half size of clients_size
        for idx in np.nonzero(swap_flag)[0]:
            # Swap clients' data with index are idx and -(idx+1)
            cidx1, cidx2 = shuffle_idx[idx], shuffle_idx[-(idx + 1)]
            print("client {} and client {} swap part data".format(cidx1, cidx2))
            c1, c2 = clients[cidx1], clients[cidx2]
            labels_c1 = [y for _, y in c1.ds_train]
            c1.label_array = np.unique(labels_c1)
            labels_c2 = [y for _, y in c2.ds_train]
            c2.label_array = np.unique(labels_c2)

            if len(c1.label_array) == 0 or len(c2.label_array) == 0: return
            c1_diff, c2_diff = np.setdiff1d(c1.label_array, c2.label_array, True), \
                np.setdiff1d(c2.label_array, c1.label_array, True)
            if c1_diff.size == 0 or c2_diff.size == 0: return
            c1_swap_label, c2_swap_label = np.random.choice(c1_diff, 1)[0], np.random.choice(c2_diff, 1)[0]
            print("client {} swap label {} with client {} swap label {}".format(cidx1, c1_swap_label, cidx2,
                                                                                c2_swap_label))

            c1_labels = [y for _, y in c1.ds_train]
            c2_labels = [y for _, y in c2.ds_train]

            label_idx1 = np.where(np.array(c1_labels) == c1_swap_label)[0]
            label_idx2 = np.where(np.array(c2_labels) == c2_swap_label)[0]

            non_swap_idx1 = np.where(np.array(c1_labels) != c1_swap_label)[0]
            non_swap_idx2 = np.where(np.array(c2_labels) != c2_swap_label)[0]

            non_swap_data_c1 = CustomSubset(c1.ds_train, non_swap_idx1)
            non_swap_data_c2 = CustomSubset(c2.ds_train, non_swap_idx2)

            swap_data_c1 = CustomSubset(c1.ds_train, label_idx1)
            swap_data_c2 = CustomSubset(c2.ds_train, label_idx2)

            # 使用ConcatDataset来创建新的c1和c2数据集
            new_c1_data = CustomConcatDataset([non_swap_data_c1, swap_data_c2])
            new_c2_data = CustomConcatDataset([non_swap_data_c2, swap_data_c1])

            c1.ds_train = new_c1_data
            c2.ds_train = new_c2_data
            
            ##### test

            c1_labels_test = [y for _, y in c1.ds_test]
            c2_labels_test = [y for _, y in c2.ds_test]

            label_idx1_test = np.where(np.array(c1_labels_test) == c1_swap_label)[0]
            label_idx2_test = np.where(np.array(c2_labels_test) == c2_swap_label)[0]

            non_swap_idx1_test = np.where(np.array(c1_labels_test) != c1_swap_label)[0]
            non_swap_idx2_test = np.where(np.array(c2_labels_test) != c2_swap_label)[0]

            non_swap_data_c1_test = CustomSubset(c1.ds_test, non_swap_idx1_test)
            non_swap_data_c2_test = CustomSubset(c2.ds_test, non_swap_idx2_test)

            swap_data_c1_test = CustomSubset(c1.ds_test, label_idx1_test)
            swap_data_c2_test = CustomSubset(c2.ds_test, label_idx2_test)

            # 使用ConcatDataset来创建新的c1和c2数据集
            new_c1_data_test = CustomConcatDataset([non_swap_data_c1_test, swap_data_c2_test])
            new_c2_data_test = CustomConcatDataset([non_swap_data_c2_test, swap_data_c1_test])

            c1.ds_test = new_c1_data_test
            c2.ds_test = new_c2_data_test

            ######
            c1.refresh_dl()
            c2.refresh_dl()
    gc.collect()


"""increase: change dl only"""


shuffle_index_dict = {}


def increase_data(r, clients):
    processing_round = [0, 12, 25, 38]
    rate = [1 / 4, 1 / 2, 3 / 4, 1.0]

    if r == 0:
        # Shuffle the train data
        for idx, c in enumerate(clients):
            c_train_idx = np.arange(len(c.ds_train))
            np.random.shuffle(c_train_idx)
            shuffle_index_dict[idx] = c_train_idx

    if r in processing_round:
        release_rate = rate[processing_round.index(r)]
        print('>Round {:3d}, {:.1%} training data release.'.format(r, release_rate))
        for idx, c in enumerate(clients):
            # Calculate new train size
            train_size = ceil(len(c.ds_train) * release_rate)
            release_index = shuffle_index_dict[idx][:train_size]
            c_subset = CustomSubset_incre(c.ds_train, release_index)
            c.ldr_train = DataLoader(c_subset, batch_size=c.local_bs, shuffle=True, drop_last=True)


index_dict_bylabels = {}


def increase_by_label(r, clients):
    processing_round = [0, 50, 100, 150]
    rate = [1 / 4, 1 / 2, 3 / 4, 1.0]

    if r == 0 and len(index_dict_bylabels) == 0:
        # Shuffle the train data by labels
        for c_idx, c in enumerate(clients):
            # Get all unique labels present in this client's train dataset
            c_uni_labels = list(set([y for _, y in c.ds_train]))
            c_all_labels = [y for _, y in c.ds_train]
            random.shuffle(c_uni_labels)
            idx_by_label = []
            for c_uni_label in c_uni_labels:
                idx_k = np.where(np.array(c_all_labels) == c_uni_label)[0]
                idx_by_label.append(idx_k)

            np.random.shuffle(idx_by_label)

            if len(c_uni_labels) > 1:  # in case there is only one label in this client's train dataset
                index_dict_bylabels[c_idx] = np.concatenate(idx_by_label, axis=0)
            else:
                index_dict_bylabels[c_idx] = idx_by_label[0]

    if r in processing_round:
        print('>Round {:3d}, releasing data by label from each client.'.format(r))

        for c_idx, c in enumerate(clients):
            release_rate = rate[processing_round.index(r)]

            train_size = ceil(len(c.ds_train) * release_rate)
            release_index = index_dict_bylabels[c_idx][:train_size]
            c_subset = CustomSubset_incre(c.ds_train, release_index)
            c.ldr_train = DataLoader(c_subset, batch_size=c.local_bs, shuffle=True, drop_last=True)


def rotate_data(clients, p):
    m = max(int(len(clients) * p), 1)
    idxs_users = np.random.choice(range(len(clients)), m, replace=False)
    for idx in idxs_users:
        print("client {} rotate data".format(idx))
        rotated_train_ds = RotatedDataset(clients[idx].ds_train, 90)
        rotated_test_ds = RotatedDataset(clients[idx].ds_test, 90)
        clients[idx].ds_train = rotated_train_ds
        clients[idx].ds_test = rotated_test_ds
        clients[idx].refresh_dl()
