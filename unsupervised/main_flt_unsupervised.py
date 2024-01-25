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
from tqdm import tqdm

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

"""FLT"""


def min_matching_distance(center_0, center_1):
    if len(center_0) < len(center_1):
        center_small = center_0
        center_big = center_1
    else:
        center_small = center_1
        center_big = center_0

    distance = np.inf
    if len(center_small) > 0:
        s = set(range(len(center_big)))
        for p in multiset_permutations(s):
            summation = 0

            for i in range(len(center_small)):
                summation = summation + (np.linalg.norm(center_small[i] - center_big[p][i]) ** 2)

            dist = np.sqrt(summation) / len(center_small)
            if dist < distance:
                distance = dist

    return distance


def encoder_model_capsul(args):
    '''
    encapsulates encoder model components
    '''
    if args.dataset in ['cifar10', 'cifar100', 'cifar110', 'cinic10']:
        ae_model = ConvAE().to(args.device)

        args.latent_dim = 64 * 4 * 4
        # loss
        criterion = nn.MSELoss()
    elif args.dataset in ['stl10']:
        ae_model = ConvAE_STL10().to(args.device)
        args.latent_dim = 128 * 6 * 6
        criterion = nn.MSELoss()
    else:
        args.latent_dim = 128
        ae_model = ConvAutoencoder().to(args.device)
        # loss
        criterion = nn.BCELoss()
    print(ae_model)
    # summary(ae_model)

    # ae_optimizer = optim.SGD(ae_model.parameters(), lr=0.001, momentum=0.9)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

    # Decay LR by a factor of x*gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(ae_optimizer, step_size=10, gamma=0.5)

    ae_model_dict = {
        'model': ae_model,
        'opt': ae_optimizer,
        'criterion': criterion,
        'scheduler': exp_lr_scheduler
    }
    return ae_model_dict


center_dict = {}
ae_model_dict = encoder_model_capsul(args)
pretrain_ds_list = train_ds_list
ae_models_list = []
num_center = args.nclusters
centers = []
for idx in range(args.num_users):
    user_dataset_train = pretrain_ds_list[idx]
    pretrain_dl = DataLoader(user_dataset_train, batch_size=64, shuffle=True, drop_last=True)
    pre_train_round = args.local_ep
    ae_client = copy.deepcopy(ae_model_dict)
    # pretrain Autoencoder
    for _ in range(pre_train_round):
        ae_client['model'].train()
        current_loss = 0.0
        for images, _ in pretrain_dl:
            images = images.to(args.device)
            # zero the parameter gradients
            ae_client['opt'].zero_grad()
            outputs, _ = ae_client['model'](images, return_comp=True)
            loss = ae_client['criterion'](outputs, images)
            loss.backward()
            ae_client['opt'].step()
            # statistics
            current_loss += loss.item() * images.size(0)
        ae_client['scheduler'].step()
        print(f'Client {idx} Pretrain Loss: {current_loss / len(user_dataset_train)}')
    embedding_list = []
    labels_list = []
    with torch.no_grad():
        for _, (image, label) in enumerate(
                tqdm(pretrain_dl)):
            image = image.to(args.device)
            labels_list.append(label)
            _, embedding = ae_client['model'](image, return_comp=True)
            embedding_list.append(embedding.cpu().detach().numpy())
    ae_embedding_np = np.concatenate(embedding_list, axis=0)
    # ae_labels_np = np.array(labels_list)  # useless
    embedding = ae_embedding_np

    kmeans = KMeans(n_clusters=num_center, random_state=args.seed, n_init=10).fit(embedding)
    centers.extend(kmeans.cluster_centers_.tolist())

    center_dict[idx] = kmeans.cluster_centers_

umap_reducer = umap.UMAP(n_components=2, random_state=args.seed)
umap_reducer.fit(np.reshape(centers, (-1, args.latent_dim)))

clustering_matrix_soft = np.zeros((args.num_users, args.num_users))
clustering_matrix = np.zeros((args.num_users, args.num_users))

c_dict = {}
for idx in range(args.num_users):
    c_dict[idx] = umap_reducer.transform(center_dict[idx])

for idx0 in tqdm(range(args.num_users)):
    c0 = c_dict[idx0]
    for idx1 in range(args.num_users):
        c0 = c_dict[idx0]
        c1 = c_dict[idx1]

        distance = min_matching_distance(c0, c1)

        clustering_matrix_soft[idx0][idx1] = distance

        if distance < 1:
            clustering_matrix[idx0][idx1] = 1
        else:
            clustering_matrix[idx0][idx1] = 0

# """clustering_matrix to cluster"""
#
#
# def dfs(current, visited, clustering_matrix):
#     stack = [current]
#     cluster = []
#
#     while stack:
#         node = stack.pop()
#         if node not in visited:
#             visited.add(node)
#             cluster.append(node)
#
#             # 查找与当前节点连接的所有节点
#             neighbors = [i for i, val in enumerate(clustering_matrix[node]) if val == 1 and i not in visited]
#             stack.extend(neighbors)
#
#     return cluster
#
#
# def get_clusters(clustering_matrix):
#     visited = set()
#     clusters = []
#
#     for user in range(len(clustering_matrix)):
#         if user not in visited:
#             cluster = dfs(user, visited, clustering_matrix)
#             clusters.append(cluster)
#
#     return clusters

def partition_clusters(clustering_matrix, args, nr_clusters=5, method='ward', metric='euclidean', plotting=False):
    # Clustering with linkage
    # Gives back linkage matrix after hierarchical clustering
    Y = sch.linkage(clustering_matrix, method=method, metric=metric)

    # Calculate cluster membership
    # fcluster flattens out dendograms to the specified nr_clusters
    cluster_memberships = sch.fcluster(Y, t=nr_clusters, criterion='maxclust') # ith element in this array is the cluster for i

    # Build cluster id to client id user dict
    cluster_user_dict = { i: np.where(cluster_memberships == i)[0] for i in range(1, nr_clusters+1)}

    clusters = [cluster_user_dict[i] for i in range(1, nr_clusters + 1)]

    return clusters

cnt = args.num_users
for r in range(1):
    print(f'Round {r}')
    clients_idxs = np.arange(cnt)
    # clients_idxs = np.arange(10)
    for idx in clients_idxs:
        print(f'Client {idx}, Labels: {traindata_cls_counts[idx]}')

    distance_matrix = clustering_matrix_soft
    print('')
    print('Distance Matrix')
    print(distance_matrix.tolist())

    # hierarchical clustering
    clusters = partition_clusters(clustering_matrix, args, nr_clusters=args.nclusters)
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
