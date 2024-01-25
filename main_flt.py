import numpy as np

import copy
import os
import gc
import pickle
import umap

import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sympy.utilities.iterables import multiset_permutations
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import *
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
    if args.dataset in ['cifar10', 'cifar100', 'cifar110', 'cinic10', 'tiny']:
        ae_model = ConvAE().to(args.device)

        args.latent_dim = 64 * 4 * 4
        # loss
        criterion = nn.MSELoss()
    # if args.target_dataset in ['CIFAR10', 'CIFAR100', 'CIFAR110']:
    #     # ae_model = ConvAutoencoderCIFAR(latent_size).to(args.device)
    #     args.num_hiddens = 128
    #     args.num_residual_hiddens = 32
    #     args.num_residual_layers = 2
    #     args.latent_dim = 64
    #
    #     ae_model = ConvAutoencoderCIFARResidual(args.num_hiddens, args.num_residual_layers,
    #                                             args.num_residual_hiddens, args.latent_dim).to(args.device)
    #
    #     # loss
    #     criterion = nn.MSELoss()
    # elif args.dataset in ['stl10']:
    #     ae_model = ConvAE_STL10().to(args.device)
    #     args.latent_dim = 128 * 6 * 6
    #     criterion = nn.MSELoss()
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
    pretrain_dl = DataLoader(user_dataset_train, batch_size=args.local_bs, shuffle=True, drop_last=True)
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
    if len(embedding_list) == 1:
        ae_embedding_np = embedding_list[0]
    else:
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

    clusters = [cluster_user_dict[i].tolist() for i in range(1, nr_clusters + 1)]

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

##### TSNE plot
center_list = [center_dict[i].flatten() for i in range(args.num_users)]
ground_truth = [[i for i in range(0, 20)], [i for i in range(20, 40)], [i for i in range(40, 60)],
                [i for i in range(60, 80)], [i for i in range(80, 100)]]

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(np.array(center_list))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  '#8c564b',  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

labels = ['Label: {0, 1}', 'Label: {2, 3}', 'Label: {4, 5}', 'Label: {6, 7}', 'Label: {8, 9}']

plt.figure(figsize=(8, 6))

for i, cluster in enumerate(ground_truth):
    if i >= len(colors):
        print("Warning: Not enough colors for all clusters, some clusters will have the same color.")
        break

    cluster_points = tsne_results[[index for index in cluster], :]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=labels[i])

plt.legend(fontsize='large')
plt.savefig('../tsne_pathological_flt.png')

###################################### Federation

loss_train = []

init_tracc_pr = []  # initial train accuracy for each round
final_tracc_pr = []  # final train accuracy for each round

init_tacc_pr = []  # initial test accuarcy for each round
final_tacc_pr = []  # final test accuracy for each round

init_tloss_pr = []  # initial test loss for each round
final_tloss_pr = []  # final test loss for each round

clients_best_acc = [0 for _ in range(args.num_users)]
w_locals, loss_locals = [], []

init_local_tacc = []  # initial local test accuracy at each round
final_local_tacc = []  # final local test accuracy at each round

init_local_tloss = []  # initial local test loss at each round
final_local_tloss = []  # final local test loss at each round

ckp_avg_tacc = []
ckp_avg_best_tacc = []

w_glob_per_cluster = [copy.deepcopy(initial_state_dict) for _ in range(len(clusters))]

users_best_acc = [0 for _ in range(args.num_users)]
best_glob_acc = [0 for _ in range(len(clusters))]

print_flag = False
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

        loss, acc = clients[idx].eval_test()

        init_local_tacc.append(acc)
        init_local_tloss.append(loss)

        loss = clients[idx].train(is_print=False)

        loss_locals.append(copy.deepcopy(loss))

        loss, acc = clients[idx].eval_test()

        if acc > clients_best_acc[idx]:
            clients_best_acc[idx] = acc

        final_local_tacc.append(acc)
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
        _, acc = eval_test(net_glob, args, test_dl_global)
        if acc > best_glob_acc[k]:
            best_glob_acc[k] = acc

            # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    avg_init_tacc = sum(init_local_tacc) / len(init_local_tacc)
    avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)
    avg_final_tacc = sum(final_local_tacc) / len(final_local_tacc)

    print('## END OF ROUND ##')
    template = 'Average Train loss {:.3f}'
    print(template.format(loss_avg))

    template = "AVG Init Test Loss: {:.3f}, AVG Init Test Acc: {:.3f}"
    print(template.format(avg_init_tloss, avg_init_tacc))

    template = "AVG Final Test Loss: {:.3f}, AVG Final Test Acc: {:.3f}"
    print(template.format(avg_final_tloss, avg_final_tacc))

    print_flag = True

    if print_flag:
        print('--- PRINTING ALL CLIENTS STATUS ---')
        current_acc = []
        for k in range(args.num_users):
            loss, acc = clients[k].eval_test()
            current_acc.append(acc)

            template = ("Client {:3d}, labels {}, count {}, best_acc {:3.3f}, current_acc {:3.3f} \n")
            print(template.format(k, traindata_cls_counts[k], clients[k].get_count(),
                                  clients_best_acc[k], current_acc[-1]))

        template = ("Round {:1d}, Avg current_acc {:3.3f}, Avg best_acc {:3.3f}")
        print(template.format(iteration + 1, np.mean(current_acc), np.mean(clients_best_acc)))

        ckp_avg_tacc.append(np.mean(current_acc))
        ckp_avg_best_tacc.append(np.mean(clients_best_acc))

    print('----- Analysis End of Round -------')
    for idx in idxs_users:
        print(f'Client {idx}, Count: {clients[idx].get_count()}, Labels: {traindata_cls_counts[idx]}')

    print('')
    print(f'Clusters {idx_clusters_round}')
    print('')

    loss_train.append(loss_avg)

    init_tacc_pr.append(avg_init_tacc)
    init_tloss_pr.append(avg_init_tloss)

    final_tacc_pr.append(avg_final_tacc)
    final_tloss_pr.append(avg_final_tloss)

    # break;
    ## clear the placeholders for the next round
    loss_locals.clear()
    init_local_tacc.clear()
    init_local_tloss.clear()
    final_local_tacc.clear()
    final_local_tloss.clear()

    ## calling garbage collector
    gc.collect()

############################### Saving Training Results
# with open(path+str(args.trial)+'_loss_train.npy', 'wb') as fp:
#     loss_train = np.array(loss_train)
#     np.save(fp, loss_train)

# with open(path+str(args.trial)+'_init_tacc_pr.npy', 'wb') as fp:
#     init_tacc_pr = np.array(init_tacc_pr)
#     np.save(fp, init_tacc_pr)

# with open(path+str(args.trial)+'_init_tloss_pr.npy', 'wb') as fp:
#     init_tloss_pr = np.array(init_tloss_pr)
#     np.save(fp, init_tloss_pr)

# with open(path+str(args.trial)+'_final_tacc_pr.npy', 'wb') as fp:
#     final_tacc_pr = np.array(final_tacc_pr)
#     np.save(fp, final_tacc_pr)

# with open(path+str(args.trial)+'_final_tloss_pr.npy', 'wb') as fp:
#     final_tloss_pr = np.array(final_tloss_pr)
#     np.save(fp, final_tloss_pr)

# with open(path+str(args.trial)+'_best_glob_w.pt', 'wb') as fp:
#     torch.save(best_glob_w, fp)
############################### Printing Final Test and Train ACC / LOSS
test_loss = []
test_acc = []
train_loss = []
train_acc = []

for idx in range(args.num_users):
    loss, acc = clients[idx].eval_test()

    test_loss.append(loss)
    test_acc.append(acc)

    loss, acc = clients[idx].eval_train()

    train_loss.append(loss)
    train_acc.append(acc)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_acc) / len(test_acc)

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_acc) / len(train_acc)

print(f'init_tacc_pr: {init_tacc_pr}')
print('')
print(f'init_tloss_pr: {init_tloss_pr}')
print('')
print(f'final_tacc_pr: {final_tacc_pr}')
print('')
print(f'final_tloss_pr: {final_tloss_pr}')
print('')

print(f'Best Clients AVG Acc: {np.mean(clients_best_acc)}')

for jj in range(len(clusters)):
    print(f'Cluster {jj}, Best Glob Acc {best_glob_acc[jj]:.3f}')

print(f'Average Best Glob Acc {np.mean(best_glob_acc[0:len(clusters)]):.3f}')
print(f'max final_tacc_pr: {max(final_tacc_pr)}')
print(f'ckp_avg_tacc: {ckp_avg_tacc}')

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')
