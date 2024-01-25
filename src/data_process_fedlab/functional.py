import copy
import random

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


def split_indices(num_cumsum, rand_perm):
    """Splice the sample index list given number of each client.

    Args:
        num_cumsum (np.ndarray): Cumulative sum of sample number for each client.
        rand_perm (list): List of random sample index.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_dict = dict(client_indices_pairs)
    return client_dict


def balance_split(num_clients, num_samples):
    """Assign same sample sample for each client.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(
        int)
    return client_sample_nums


def lognormal_unbalance_split(num_clients, num_samples, unbalance_sgm):
    """Assign different sample number for each client using Log-Normal distribution.

    Sample numbers for clients are drawn from Log-Normal distribution.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.
        unbalance_sgm (float): Log-normal variance. When equals to ``0``, the partition is equal to :func:`balance_partition`.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    if unbalance_sgm != 0:
        client_sample_nums = np.random.lognormal(mean=np.log(num_samples_per_client),
                                                 sigma=unbalance_sgm,
                                                 size=num_clients)
        client_sample_nums = (
                client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0

        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(num_clients):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
    else:
        client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)

    return client_sample_nums


def dirichlet_unbalance_split(num_clients, num_samples, alpha):
    """Assign different sample number for each client using Dirichlet distribution.

    Sample numbers for clients are drawn from Dirichlet distribution.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.
        alpha (float): Dirichlet concentration parameter

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    min_size = 0
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * num_samples)

    client_sample_nums = (proportions * num_samples).astype(int)
    return client_sample_nums


def homo_partition(client_sample_nums, num_samples):
    """Partition data indices in IID way given sample numbers for each clients.

    Args:
        client_sample_nums (numpy.ndarray): Sample numbers for each clients.
        num_samples (int): Number of samples.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    rand_perm = np.random.permutation(num_samples)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    client_dict = split_indices(num_cumsum, rand_perm)
    return client_dict


def hetero_dir_partition(targets, num_clients, num_classes, dir_alpha, min_require_size=None):
    """

    Non-iid partition based on Dirichlet distribution. The method is from "hetero-dir" partition of
    `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
    and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_.

    This method simulates heterogeneous partition for which number of data points and class
    proportions are unbalanced. Samples will be partitioned into :math:`J` clients by sampling
    :math:`p_k \sim \\text{Dir}_{J}({\\alpha})` and allocating a :math:`p_{p,j}` proportion of the
    samples of class :math:`k` to local client :math:`j`.

    Sample number for each client is decided in this function.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    if min_require_size is None:
        min_require_size = num_classes

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(dir_alpha, num_clients))
            # Balance
            proportions = np.array(
                [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                 zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_dict = dict()
    for cid in range(num_clients):
        np.random.shuffle(idx_batch[cid])
        client_dict[cid] = np.array(idx_batch[cid])

    return client_dict


def shards_partition(targets, num_clients, num_shards):
    """Non-iid partition used in FedAvg `paper <https://arxiv.org/abs/1602.05629>`_.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_shards (int): Number of shards in partition.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    size_shard = int(num_samples / num_shards)
    if num_samples % num_shards != 0:
        warnings.warn("warning: length of dataset isn't divided exactly by num_shards. "
                      "Some samples will be dropped.")

    shards_per_client = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn("warning: num_shards isn't divided exactly by num_clients. "
                      "Some shards will be dropped.")

    indices = np.arange(num_samples)
    # sort sample indices according to labels
    indices_targets = np.vstack((indices, targets))
    indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    sorted_indices = indices_targets[0, :]

    # permute shards idx, and slice shards_per_client shards for each client
    rand_perm = np.random.permutation(num_shards)
    num_client_shards = np.ones(num_clients) * shards_per_client
    # sample index must be int
    num_cumsum = np.cumsum(num_client_shards).astype(int)
    # shard indices for each client
    client_shards_dict = split_indices(num_cumsum, rand_perm)

    # map shard idx to sample idx for each client
    client_dict = dict()
    for cid in range(num_clients):
        shards_set = client_shards_dict[cid]
        current_indices = [
            sorted_indices[shard_id * size_shard: (shard_id + 1) * size_shard]
            for shard_id in shards_set]
        client_dict[cid] = np.concatenate(current_indices, axis=0)

    return client_dict


def client_inner_dirichlet_partition(targets, num_clients, num_classes, dir_alpha,
                                     client_sample_nums, verbose=True):
    """Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.
    It's different from :func:`hetero_dir_partition`.

    Args:
        targets (list or numpy.ndarray): Sample targets.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        client_sample_nums (numpy.ndarray): A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    rand_perm = np.random.permutation(targets.shape[0])
    targets = targets[rand_perm]

    class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                       size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                      range(num_clients)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict


def label_skew_quantity_based_partition(targets, num_clients, num_classes, major_classes_num):
    """Label-skew:quantity-based partition.

    For details, please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Args:
        targets (List or np.ndarray): Labels od dataset.
        num_clients (int): Number of clients.
        num_classes (int): Number of unique classes.
        major_classes_num (int): Number of classes for each client, should be less then ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    idx_batch = [np.ndarray(0, dtype=np.int64) for _ in range(num_clients)]
    # only for major_classes_num < num_classes.
    # if major_classes_num = num_classes, it equals to IID partition
    times = [0 for _ in range(num_classes)]
    contain = []
    for cid in range(num_clients):
        current = [cid % num_classes]
        times[cid % num_classes] += 1
        j = 1
        while j < major_classes_num:
            ind = np.random.randint(num_classes)
            if ind not in current:
                j += 1
                current.append(ind)
                times[ind] += 1
        contain.append(current)

    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        split = np.array_split(idx_k, times[k])
        ids = 0
        for cid in range(num_clients):
            if k in contain[cid]:
                idx_batch[cid] = np.append(idx_batch[cid], split[ids])
                ids += 1

    client_dict = {cid: idx_batch[cid] for cid in range(num_clients)}
    return client_dict


def fcube_synthetic_partition(data):
    """Feature-distribution-skew:synthetic partition.

    Synthetic partition for FCUBE dataset. This partition is from `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Args:
        data (np.ndarray): Data of dataset :class:`FCUBE`.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    num_clients = 4
    client_indices = [[] for _ in range(num_clients)]
    for idx, sample in enumerate(data):
        p1, p2, p3 = sample
        if (p1 > 0 and p2 > 0 and p3 > 0) or (p1 < 0 and p2 < 0 and p3 < 0):
            client_indices[0].append(idx)
        elif (p1 > 0 and p2 > 0 and p3 < 0) or (p1 < 0 and p2 < 0 and p3 > 0):
            client_indices[1].append(idx)
        elif (p1 > 0 and p2 < 0 and p3 > 0) or (p1 < 0 and p2 > 0 and p3 < 0):
            client_indices[2].append(idx)
        else:
            client_indices[3].append(idx)
    client_dict = {cid: np.array(client_indices[cid]).astype(int) for cid in range(num_clients)}
    return client_dict


def samples_num_count(client_dict, num_clients):
    """Return sample count for all clients in ``client_dict``.

    Args:
        client_dict (dict): Data partition result for different clients.
        num_clients (int): Total number of clients.

    Returns:
        pandas.DataFrame

    """
    client_samples_nums = [[cid, client_dict[cid].shape[0]] for cid in
                           range(num_clients)]
    client_sample_count = pd.DataFrame(data=client_samples_nums,
                                       columns=['client', 'num_samples']).set_index('client')
    return client_sample_count


def noniid_slicing(dataset, num_clients, num_shards):
    """Slice a dataset for non-IID.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to slice.
        num_clients (int):  Number of client.
        num_shards (int): Number of shards.
    
    Notes:
        The size of a shard equals to ``int(len(dataset)/num_shards)``.
        Each client will get ``int(num_shards/num_clients)`` shards.

    Returns：
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    total_sample_nums = len(dataset)
    size_of_shards = int(total_sample_nums / num_shards)
    if total_sample_nums % num_shards != 0:
        warnings.warn(
            "warning: the length of dataset isn't divided exactly by num_shard.some samples will be dropped."
        )
    # the number of shards that each one of clients can get
    shard_pc = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shard isn't divided exactly by num_clients. some samples will be dropped."
        )

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

    labels = np.array(dataset.targets)
    idxs = np.arange(total_sample_nums)

    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]

    # assign
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i],
                 idxs[rand * size_of_shards:(rand + 1) * size_of_shards]),
                axis=0)

    return dict_users


def random_slicing(dataset, num_clients):
    """Slice a dataset randomly and equally for IID.

    Args：
        dataset (torch.utils.data.Dataset): a dataset for slicing.
        num_clients (int):  the number of client.

    Returns：
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = list(
            np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def noniid_slicing_with_dirichlet(dataset, num_clients, num_classes, alpha):
    min_size = 0
    min_require_size = 10
    K = num_classes

    y_train = np.array(dataset.targets)
    N = y_train.shape[0]
    # np.random.seed(2021)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def noniid_slicing_label_skew(dataset, num_clients, num_classes, major_classes_num):
    num = major_classes_num
    n_parties = num_clients
    K = num_classes
    y_train = np.array(dataset.targets)

    print(f'K: {K}')
    if num == 10:
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(10):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, n_parties)
            for j in range(n_parties):
                net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
    else:
        times = [0 for i in range(K)]
        contain = []
        for i in range(n_parties):
            current = [i % K]
            times[i % K] += 1
            j = 1
            while (j < num):
                ind = random.randint(0, K - 1)
                if (ind not in current):
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1

    for i in range(n_parties):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


def noniid_label_100(dataset, num_clients, major_classes_num):
    y_train = np.array(dataset.targets)
    n_parties = num_clients
    print('Modified Non-IID partitioning')
    num = major_classes_num
    K = 100

    if num == 10:
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(10):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, n_parties)
            for j in range(n_parties):
                net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
    else:
        times = [0 for i in range(K)]
        contain = []

        # aa = np.random.randint(low=0, high=K, size=num)
        aa = np.random.choice(np.arange(K), size=num, replace=False)
        remain = np.delete(np.arange(K), aa)
        # print(f'Client 0 , {len(aa)}')
        # print(f'Unique a {len(np.unique(aa))}')
        # print(f'Unique remain {len(np.unique(remain))}')
        contain.append(copy.deepcopy(aa.tolist()))
        for el in aa:
            times[el] += 1

        for i in range(n_parties - 1):
            x = np.random.randint(low=int(np.ceil(K / 2)), high=K)
            y = np.random.randint(low=0, high=int(K / 4) + 1)

            rand = np.random.choice([0, 1, 2], size=1, replace=False)
            # print(rand)
            if rand == 0 or rand == 1:
                s = int(np.ceil((x / K) * num))
                if s == num and rand == 0:
                    s = s - int(np.ceil(0.05 * num))
            elif rand == 2:
                s = int(np.ceil((y / K) * num))

            labels = np.random.choice(aa, size=s, replace=False).tolist()
            # print(f'Client {i} , {len(labels)}, S {s}')
            # print(labels)
            labels.extend(np.random.choice(remain, size=(num - s), replace=False).tolist())
            # print(f'Client {i+1} , {len(labels)}')
            # ccc = np.unique(labels)
            # print(f'Client {i+1} , {len(ccc)}')

            for el in labels:
                times[el] += 1
            contain.append(labels)
            # print(len(labels))

        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            # print(f'{i}: {times[i]}')
            split = np.array_split(idx_k, times[i])
            # print(f'len(split) {len(split)}, times[i] {times[i]}')
            ids = 0
            for j in range(n_parties):
                # print(f'Client {i}, {len(contain[j])}')
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1
    for i in range(n_parties):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


def pathological_slicing(dataset, num_clients, num_classes, labels_per_group=2):
    assert num_classes % labels_per_group == 0
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}
    num_groups = int(num_classes / labels_per_group)  # 标签组的数量
    num_clients_per_group = int(num_clients / num_groups)  # 每个标签组包含的客户端数量

    idx_per_group = [[] for _ in range(num_groups)]
    current_client_id = 0
    for group_id in range(num_groups):
        for label_id in range(labels_per_group):
            idx_per_group[group_id].append(np.where(y_train == group_id * labels_per_group + label_id)[0])
        idx_per_group[group_id] = np.random.permutation(np.concatenate(idx_per_group[group_id]))
        idx_per_client = np.array_split(idx_per_group[group_id], num_clients_per_group)

        for client_id_in_group in range(num_clients_per_group):
            net_dataidx_map[current_client_id] = copy.deepcopy(idx_per_client[client_id_in_group])
            current_client_id += 1

    for i in range(num_clients):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


def pathological_slicing_cifar100(dataset, num_clients, num_classes):
    group1_labels = [4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 6, 7, 14, 18, 24, 3, 42, 43, 88, 97, 15, 19, 21, 31, 38, 34, 63, 64, 66, 75, 26, 44, 77, 79, 99, 2, 11, 35, 46, 98, 27, 29, 45, 78, 93, 36, 50, 65, 74, 80]
    group2_labels = [9, 10, 16, 28, 61, 0, 51, 53, 57, 83, 22, 39, 40, 86, 87, 5, 20, 25, 84, 94, 12, 17, 37, 68, 76, 8, 13, 48, 58, 90, 41, 69, 81, 85, 89, 23, 33, 49, 60, 71, 54, 62, 70, 82, 92, 47, 52, 56, 59, 96]
    assert len(group1_labels) + len(group2_labels) == num_classes
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}

    # Create two groups based on provided labels
    idx_group1 = []
    idx_group2 = []

    # Assign each label to its respective group
    for label in group1_labels:
        idx_group1.append(np.where(y_train == label)[0])
    for label in group2_labels:
        idx_group2.append(np.where(y_train == label)[0])

    idx_group1 = np.random.permutation(np.concatenate(idx_group1))
    idx_group2 = np.random.permutation(np.concatenate(idx_group2))

    # Calculate the number of clients per group
    num_clients_per_group = num_clients // 2

    # Split each group among the clients
    idx_per_client_group1 = np.array_split(idx_group1, num_clients_per_group)
    idx_per_client_group2 = np.array_split(idx_group2, num_clients_per_group)

    for i in range(num_clients_per_group):
        net_dataidx_map[i] = copy.deepcopy(idx_per_client_group1[i])
        net_dataidx_map[i + num_clients_per_group] = copy.deepcopy(idx_per_client_group2[i])

    for i in range(num_clients):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


def pathological_slicing_2(dataset, num_clients, groups):
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}
    num_groups = len(groups)  # 标签组的数量
    num_clients_per_group = int(num_clients / num_groups)  # 每个标签组包含的客户端数量

    idx_per_group = [[] for _ in range(num_groups)]
    current_client_id = 0
    for group_id, labels in enumerate(groups):
        for label_id in labels:
            idx_per_group[group_id].append(np.where(y_train == label_id)[0])
        idx_per_group[group_id] = np.random.permutation(np.concatenate(idx_per_group[group_id]))
        idx_per_client = np.array_split(idx_per_group[group_id], num_clients_per_group)

        for client_id_in_group in range(num_clients_per_group):
            net_dataidx_map[current_client_id] = copy.deepcopy(idx_per_client[client_id_in_group])
            current_client_id += 1

    for i in range(num_clients):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map

def pathological_slicing_3(dataset, num_clients, num_classes=10):
    assert num_classes == 10  # 确保使用10个类别
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}

    # 更新分组策略：第一组为标签0, 1, 8, 9，第二组为标签3, 4, 5, 6, 7
    groups = [[0, 1, 8, 9], [3, 4, 5, 6, 7]]
    num_groups = len(groups)
    num_clients_per_group = int(num_clients / num_groups)  # 每个标签组包含的客户端数量

    idx_per_group = [[] for _ in range(num_groups)]
    current_client_id = 0

    for group_id, labels in enumerate(groups):
        for label_id in labels:
            idx_per_group[group_id].append(np.where(y_train == label_id)[0])
        idx_per_group[group_id] = np.random.permutation(np.concatenate(idx_per_group[group_id]))
        idx_per_client = np.array_split(idx_per_group[group_id], num_clients_per_group)

        for client_id_in_group in range(num_clients_per_group):
            net_dataidx_map[current_client_id] = copy.deepcopy(idx_per_client[client_id_in_group])
            current_client_id += 1

    for i in range(num_clients):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map



def single_label_slicing(dataset, num_clients, num_classes):
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}
    for i in range(num_clients):
        idx_k = np.where(y_train == i % num_classes)[0]
        np.random.shuffle(idx_k)
        net_dataidx_map[i] = idx_k

    return net_dataidx_map


def plot_label_distribution(client_number, class_num, net_dataidx_map, y_complete):
    heat_map_data = np.zeros((class_num, client_number))

    for client_idx in range(client_number):
        idxx = net_dataidx_map[client_idx]

        valuess, counts = np.unique([y_complete[i] for i in idxx], return_counts=True)
        for (i, j) in zip(valuess, counts):
            heat_map_data[i][int(client_idx)] = j / len(idxx)

    plt.figure()
    fig_dims = (30, 10)
    sns.heatmap(heat_map_data, linewidths=0.05, cmap="YlGnBu", cbar=True)
    plt.xlabel('Client number')
    plt.title("label distribution")

    fig = plt.gcf()
    plt.show()


def plot_partitions(partitions, y_complete):
    # 从partitions和cifar10_full中获取每个客户端分配到的标签
    data = {'client': [], 'label': []}
    for client, indices in partitions.items():
        for index in indices:
            data['client'].append(client)
            data['label'].append(y_complete[index])

    # 转换为pandas DataFrame
    df = pd.DataFrame(data)

    # 计算每个客户端分配到的每个标签的数量
    df = df.groupby(['client', 'label']).size().reset_index(name='count')

    # 转换为适合绘制堆叠的条形图的格式
    df = df.pivot(index='client', columns='label', values='count').fillna(0).astype(int)
    col_names = [f'class{i}' for i in range(df.columns.max() + 1)]
    df.columns = col_names

    # 绘制堆叠的条形图
    ax = df.plot.barh(stacked=True, figsize=(10, 6))

    # 设置标题和轴标签
    plt.title('Number of Samples of Each Class Assigned to Each Client')
    plt.xlabel('Sample Num')
    plt.ylabel('Client')

    # 设置图例
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 显示图
    plt.show()
    plt.savefig('../save_results/partition.png')
