import sys

import numpy as np

import copy
import os
import gc
import pickle

import torch
from sklearn.cluster import DBSCAN
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms

sys.path.append('../')  # 添加上一个文件夹的路径

from src.models import *
from src.models.GANs import GAN
from src.models.autoencoders import ConvAE, ConvAE_STL10
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
traindata_cls_ratio = {}

budget = 20
for i in range(args.num_users):
    total_sum = sum(list(traindata_cls_counts[i].values()))
    base = 1 / len(list(traindata_cls_counts[i].values()))
    temp_ratio = {}
    for k in traindata_cls_counts[i].keys():
        ss = traindata_cls_counts[i][k] / total_sum
        temp_ratio[k] = (traindata_cls_counts[i][k] / total_sum)
        if ss >= (base + 0.05):
            temp_ratio[k] = traindata_cls_counts[i][k]

    sub_sum = sum(list(temp_ratio.values()))
    for k in temp_ratio.keys():
        temp_ratio[k] = (temp_ratio[k] / sub_sum) * budget

    round_ratio = round_to(list(temp_ratio.values()), budget)
    cnt = 0
    for k in temp_ratio.keys():
        temp_ratio[k] = round_ratio[cnt]
        cnt += 1

    traindata_cls_ratio[i] = temp_ratio

clients = []
U_clients = []

K = args.n_basis
# K = 5
for idx in range(args.num_users):

    train_ds_local = train_ds_list[idx]
    test_ds_local = test_ds_list[idx]

    idxs_local = np.arange(len(train_ds_local.data))
    labels_local = np.array(train_ds_local.target)
    # Sort Labels Train
    idxs_labels_local = np.vstack((idxs_local, labels_local))
    idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
    idxs_local = idxs_labels_local[0, :]
    labels_local = idxs_labels_local[1, :]

    uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)

    print(f'Labels: {uni_labels}, Counts: {cnt_labels}')

    nlabels = len(uni_labels)
    cnt = 0
    U_temp = []
    for j in range(nlabels):
        local_ds1 = train_ds_local.data[idxs_local[cnt:cnt + cnt_labels[j]]]
        local_ds1 = local_ds1.reshape(cnt_labels[j], -1)
        local_ds1 = local_ds1.T
        if type(train_ds_local.target[idxs_local[cnt:cnt + cnt_labels[j]]]) == torch.Tensor:
            label1 = list(set(train_ds_local.target[idxs_local[cnt:cnt + cnt_labels[j]]].numpy()))
        else:
            label1 = list(set(train_ds_local.target[idxs_local[cnt:cnt + cnt_labels[j]]]))
        # assert len(label1) == 1

        # print(f'Label {j} : {label1}')

        if args.partition == 'noniid-labeldir':
            # print('Dir partition')
            if label1 in list(traindata_cls_ratio[idx].keys()):
                K = traindata_cls_ratio[idx][label1[0]]
            else:
                K = args.n_basis
        if K > 0:
            u1_temp, sh1_temp, vh1_temp = np.linalg.svd(local_ds1, full_matrices=False)
            u1_temp = u1_temp / np.linalg.norm(u1_temp, ord=2, axis=0)
            U_temp.append(u1_temp[:, 0:K])

        cnt += cnt_labels[j]

    # U_temp = [u1_temp[:, 0:K], u2_temp[:, 0:K]]
    U_clients.append(copy.deepcopy(np.hstack(U_temp)))

    print(f'Shape of U: {U_clients[-1].shape}')

    clients.append(Client_ClusterFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                                    args.lr, args.momentum, args.device, train_ds_local, test_ds_local))

###################################### Clustering
np.set_printoptions(precision=2)
# m = max(int(args.frac * args.num_users), 1)
# clients_idxs = np.random.choice(range(args.num_users), m, replace=False)

cnt = args.num_users
for r in range(1):
    print(f'Round {r}')
    clients_idxs = np.arange(cnt)
    # clients_idxs = np.arange(10)
    for idx in clients_idxs:
        print(f'Client {idx}, Labels: {traindata_cls_counts[idx]}')

    adj_mat = calculating_adjacency(clients_idxs, U_clients)

    distance_matrix = adj_mat
    print('')
    print('Distance Matrix')
    print(distance_matrix.tolist())

    print('')
    print("Cluster threshold")
    print(args.cluster_alpha)
    clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=args.cluster_alpha, linkage=args.linkage)
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

for iteration in range(args.rounds):

    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

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
