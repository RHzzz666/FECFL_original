import sys

import numpy as np

import copy
import os
import gc
import pickle
import umap

import torch
from sklearn.cluster import DBSCAN
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from sklearn.cluster import DBSCAN, KMeans
from sympy.utilities.iterables import multiset_permutations
import scipy.cluster.hierarchy as sch

sys.path.append('../')

from src.models import *
from src.models.GANs import GAN
from src.models.autoencoders import *
from src.fedavg import *
from src.client import *
from src.clustering import *
from src.utils import *
from datasets_models import get_datasets, init_nets

args = args_parser()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu)  ## Setting cuda on GPU
print('Using GPU: {} '.format(torch.cuda.current_device()))
print('Using Device: {} '.format(args.device))


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


path = args.savedir + args.alg + '/' + args.partition + '/' + args.dataset + '/'
mkdirs(path)

template = "Algorithm {}, Clients {}, Dataset {}, Model {}, Non-IID {}, Threshold {}, K {}, Linkage {}, LR {}, Ep {}, Rounds {}, bs {}, frac {}"

s = template.format(args.alg, args.num_users, args.dataset, args.model, args.partition, args.cluster_alpha,
                    args.n_basis, args.linkage, args.lr, args.local_ep, args.rounds, args.local_bs, args.frac)

print(s)

print(str(args))

data_set = get_datasets(args)

train_ds_list = [data_set.get_dataset(client_index, "train") for client_index in range(args.num_users)]
test_ds_list = [data_set.get_dataset(client_index, "test") for client_index in range(args.num_users)]

test_ds_global = ConcatDataset(test_ds_list)
test_dl_global = DataLoader(test_ds_global, batch_size=args.local_bs, shuffle=False)

train_datanum_list = [len(train_ds_list[client_index]) for client_index in range(args.num_users)]

traindata_cls_counts = {}
for client_id in range(args.num_users):
    unq, unq_cnt = np.unique(train_ds_list[client_id].target, return_counts=True)
    tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    traindata_cls_counts[client_id] = tmp

print(f'MODEL: {args.model}, Dataset: {args.dataset}')

users_model, net_glob, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)

print(net_glob)

total = 0
for name, param in net_glob.named_parameters():
    print(name, param.size())
    total += np.prod(param.size())
    # print(np.array(param.data.cpu().numpy().reshape([-1])))
    # print(isinstance(param.data.cpu().numpy(), np.array))
print(total)


################################# Initializing Clients

clients = []

for idx in range(args.num_users):
    train_ds_local = train_ds_list[idx]
    test_ds_local = test_ds_list[idx]

    clients.append(Client_ClusterFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                                    args.lr, args.momentum, args.device, train_ds_local, test_ds_local))

###################################### Clustering
np.set_printoptions(precision=2)

cnt = args.num_users
for r in range(1):
    print(f'Round {r}')
    clients_idxs = np.arange(cnt)
    # clients_idxs = np.arange(10)
    for idx in clients_idxs:
        print(f'Client {idx}, Labels: {traindata_cls_counts[idx]}')

    clusters = [[_ for _ in range(args.num_users)]]

    cnt += 10
    print('')
    print('Clusters: ')
    print(clusters)
    print('')
    print(f'Number of Clusters {len(clusters)}')
    print('')
    for jj in range(len(clusters)):
        print(f'Cluster {jj}: {len(clusters[jj])} Users')

clients_clust_id = {i: None for i in range(args.num_users)}
for i in range(args.num_users):
    for j in range(len(clusters)):
        if i in clusters[j]:
            clients_clust_id[i] = j
            break
print(f'Clients: Cluster_ID \n{clients_clust_id}')

###################################### Federation

loss_train = []

init_tloss_pr = []  # initial test loss for each round
final_tloss_pr = []  # final test loss for each round

w_locals, loss_locals = [], []

init_local_tloss = []  # initial local test loss at each round
final_local_tloss = []  # final local test loss at each round

w_glob_per_cluster = [copy.deepcopy(initial_state_dict) for _ in range(len(clusters))]
clients_lowest_loss = [float('inf') for _ in range(args.num_users)]


multi_center_initialization_flag = True
est_multi_center = []


def clustering_multi_center(num_users, w_local_list, multi_center_initialization_flag, est_multi_center, args):
    lst = [list(w_local_list[0][k].cpu().numpy().flatten()) for k in w_local_list[0].keys()]
    flat_list = [item for sublist in lst for item in sublist]
    model_params_length = len(flat_list)
    models_parameter_list = np.zeros((num_users, model_params_length))

    for i in range(num_users):
        model = w_local_list[i]
        lst = [list(model[k].cpu().numpy().flatten()) for k in model.keys()]
        flat_list = [item for sublist in lst for item in sublist]

        models_parameter_list[i] = np.array(flat_list).reshape(1, model_params_length)

    if multi_center_initialization_flag:
        kmeans = KMeans(n_clusters=args.nclusters, n_init=20).fit(models_parameter_list)

    else:
        kmeans = KMeans(n_clusters=args.nclusters, init=est_multi_center, n_init=1).fit(
            models_parameter_list)  # TODO: remove the best

    ind_center = kmeans.fit_predict(models_parameter_list)

    est_multi_center_new = kmeans.cluster_centers_
    unique_labels = set(ind_center)
    clusters = [list(np.where(ind_center == label)[0]) for label in unique_labels]

    return clusters, est_multi_center_new



for iteration in range(args.rounds):

    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # idxs_users = comm_users[iteration]

    print(f'###### ROUND {iteration + 1} ######')
    print(f'Clients {idxs_users}')

    idx_clusters_round = {}  # 类似group_list
    for idx in idxs_users:
        idx_cluster = clients_clust_id[idx]
        idx_clusters_round[idx_cluster] = []

    for idx in idxs_users:
        idx_cluster = clients_clust_id[idx]
        idx_clusters_round[idx_cluster].append(idx)

        clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[idx_cluster]))

        loss = clients[idx].eval_test_unsupervised()
        init_local_tloss.append(loss)

        loss = clients[idx].train_unsupervised()

        loss_locals.append(copy.deepcopy(loss))

        loss = clients[idx].eval_test_unsupervised()
        if loss < clients_lowest_loss[idx]:
            clients_lowest_loss[idx] = loss
        final_local_tloss.append(loss)

    """FedSEM"""
    w_local_list = [copy.deepcopy(client.get_state_dict()) for client in clients]
    clusters, est_multi_center = clustering_multi_center(args.num_users, w_local_list,
                                                         multi_center_initialization_flag,
                                                         est_multi_center, args)
    if multi_center_initialization_flag:
        w_glob_per_cluster = [copy.deepcopy(initial_state_dict) for _ in range(len(clusters))]

    multi_center_initialization_flag = False
    print(f'round: {iteration}, clusters: {clusters}')

    """refresh clusters"""
    clients_clust_id = {i: None for i in range(args.num_users)}
    for i in range(args.num_users):
        for j in range(len(clusters)):
            if i in clusters[j]:
                clients_clust_id[i] = j
                break
    print(f'Clients: Cluster_ID \n{clients_clust_id}')

    idx_clusters_round = {}  # 类似group_list
    for idx in idxs_users:
        idx_cluster = clients_clust_id[idx]
        idx_clusters_round[idx_cluster] = []

    for idx in idxs_users:
        idx_cluster = clients_clust_id[idx]
        idx_clusters_round[idx_cluster].append(idx)

    """"""

    total_data_points = {}
    for k in idx_clusters_round.keys():
        temp_sum = []
        for r in idx_clusters_round[k]:
            temp_sum.append(train_datanum_list[r])

        total_data_points[k] = sum(temp_sum)

    fed_avg_freqs = {}
    for k in idx_clusters_round.keys():
        fed_avg_freqs[k] = []
        for r in idx_clusters_round[k]:
            ratio = train_datanum_list[r] / total_data_points[k]
            fed_avg_freqs[k].append(copy.deepcopy(ratio))

    for k in idx_clusters_round.keys():
        w_locals = []
        for el in idx_clusters_round[k]:
            w_locals.append(copy.deepcopy(clients[el].get_state_dict()))

        ww = FedAvg(w_locals, weight_avg=fed_avg_freqs[k])
        w_glob_per_cluster[k] = copy.deepcopy(ww)
        net_glob.load_state_dict(copy.deepcopy(ww))

            # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)

    print('## END OF ROUND ##')
    template = 'Average Train loss {:.3f}'
    print(template.format(loss_avg))

    print('----- Analysis End of Round -------')
    for idx in idxs_users:
        print(f'Client {idx}, Count: {clients[idx].get_count()}, Labels: {traindata_cls_counts[idx]}')

    print('')
    print(f'Clusters {idx_clusters_round}')
    print('')

    loss_train.append(loss_avg)

    init_tloss_pr.append(avg_init_tloss)

    final_tloss_pr.append(avg_final_tloss)

    # break;
    ## clear the placeholders for the next round
    loss_locals.clear()
    init_local_tloss.clear()
    final_local_tloss.clear()

    ## calling garbage collector
    gc.collect()

############################### Printing Final Test and Train ACC / LOSS
test_loss = []
train_loss = []

for idx in range(args.num_users):
    loss = clients[idx].eval_test_unsupervised()

    test_loss.append(loss)

    loss = clients[idx].eval_train_unsupervised()

    train_loss.append(loss)

test_loss = sum(test_loss) / len(test_loss)

train_loss = sum(train_loss) / len(train_loss)

print('')
print(f'init_tloss_pr: {init_tloss_pr}')
print('')
print(f'final_tloss_pr: {final_tloss_pr}')
print('')
print(f'Lowest Clients AVG Loss: {np.mean(clients_lowest_loss)}')
print(f'min final loss: {min(final_tloss_pr)}')

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')