import sys
import copy
import os
import gc
import pickle

import torch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np


sys.path.append('../')

from src.dynamic.distribution_shift import *
from src.models import *
from src.models.autoencoders import ConvAE
from src.fedavg import *
from src.client.client_flexcfl import Client_FlexCFL
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
global_features_dict = dict()
features_list = []

if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10

for idx in range(args.num_users):
    train_ds_local = train_ds_list[idx]
    test_ds_local = test_ds_list[idx]
    clients.append(Client_FlexCFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                                  args.lr, args.momentum, args.device, train_ds_local, test_ds_local, num_classes=n_classes))

if args.shift_type == 'incremental':
    increase_data(0, clients)

"""FlexCFL"""
'''
1. initialization
2. group_cold_start
    2.1 pretrain
    2.2 EDC and clustering
    3.3 cluster_W, cluster_dW, auxiliary_global_model
3. train
    3.1 Random select clients
    3.2 Change the clients's data distribution
    3.3 Schedule clients (for example: reassign) or cold start clients, need selected clients only
        3.3.1 distribution shift detection, and redo cold start
        3.3.2 client_cold_start for newcomers: pretrain and assign a group
    3.5 Train selected clients  # IntraGroupUpdate(FedAvg){train, aggregate}
    3.6 Inter-group aggregation according to the group learning rate  
    3.9 Update the auxiliary global model. Simply average group models without weights 

'''


def group_cold_start(args, cold_clients):
    dW = []
    W_1_list = []
    for c in cold_clients:
        c.set_state_dict(copy.deepcopy(initial_state_dict))
        dW_c, W_1 = c.pre_train(args.local_ep)  # dW_c: grad_list, W_1: params dict after pretrain
        dW.append(dW_c)
        W_1_list.append(W_1)

    delta_w = np.array(dW)  # shape=(n_clients, n_params)
    # Decomposed the directions of updates to num_group of directional vectors
    svd = TruncatedSVD(n_components=args.nclusters, random_state=args.seed)
    decomp_updates = svd.fit_transform(delta_w.T)  # shape=(n_params, n_groups)
    # n_components = decomp_updates.shape[-1]

    decomposed_cossim_matrix = cosine_similarity(delta_w, decomp_updates.T)  # shape=(n_clients, n_clients)

    affinity_matrix = decomposed_cossim_matrix
    result = KMeans(args.nclusters, max_iter=20, random_state=args.seed).fit(affinity_matrix)
    cluster_labels = result.labels_
    unique_labels = set(cluster_labels)
    clusters = [list(np.where(cluster_labels == label)[0]) for label in unique_labels]
    print(clusters)

    cluster_W = []  # omega_0,g
    cluster_dW = []  # delta omega_0,g
    for cluster_id, client_list in enumerate(clusters):
        # calculate the means of cluster
        params_list = [W_1_list[c_idx] for c_idx in client_list]  # dict
        updates_list = [delta_w[c_idx] for c_idx in client_list]  # np.array
        if params_list:
            # All client have equal weight
            cluster_W.append(FedAvg(params_list))
            cluster_dW.append(np.mean(updates_list, axis=0))
        else:
            print("Error, cluster is empty")

    # auxiliary_global_model, not important
    auxiliary_global_model = FedAvg(cluster_W)

    sim_matrix = cosine_similarity(delta_w, delta_w)

    return clusters, cluster_W, cluster_dW, auxiliary_global_model, affinity_matrix, sim_matrix


def client_cold_start(args, client, client_id, clusters, cluster_dW, clients_clust_id):
    client.set_state_dict(copy.deepcopy(initial_state_dict))
    cupdate, csoln = client.pre_train(args.local_ep)

    # Calculate the cosine dissimilarity between client's update and group's update
    sim_list = []
    for cluster_id, cluster in enumerate(clusters):
        opt_updates = cluster_dW[cluster_id]
        sim = cosine_similarity([cupdate], [opt_updates])[0][0]
        sim_list.append(sim)
    new_cluster_id = np.argmax(sim_list)
    clusters[new_cluster_id].append(client_id)
    print("Client {} is assigned to group {}".format(client_id, new_cluster_id))
    clients_clust_id[client_id] = new_cluster_id


def schedule_clients(args, clients, selected_idxs, clusters, cluster_dW, clients_clust_id):
    schedule_results = None
    # 1, Redo cold start distribution shift clients
    shift_count, migration_count = 0, 0
    for cid in selected_idxs:
        client = clients[cid]
        count, shifted = client.check_distribution_shift()
        if count is not None and shifted:
            print("Client {} detects distribution shift".format(cid))
            shift_count += 1
            prev_g = clients_clust_id[cid]

            client_cold_start(args, client, cid, clusters, cluster_dW, clients_clust_id)

            new_g = clients_clust_id[cid]
            client.train_label_count = count
            client.distribution_shift = False
            if prev_g != new_g:
                migration_count += 1
                print(f'Client {cid} migrate from Group {prev_g} to Group {new_g}')
    schedule_results = {'shift': shift_count, 'migration': migration_count}

    # # 2, Cold start newcomer: pretrain and assign a group
    # for client in selected_clients:
    #     # for client in self.clients:
    #     if client.has_uplink() == False:
    #         self.client_cold_start(client, self.RAC)
    print(f'Schedule Results: {schedule_results}')


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

    clusters, cluster_W, cluster_dW, auxiliary_global_model, affinity_matrix, sim_matrix = group_cold_start(args,
                                                                                                            clients)

    distance_matrix = 1 - np.array(sim_matrix)
    print('')
    print('Distance Matrix')
    print(distance_matrix.tolist())

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
# best_glob_acc = [0 for _ in range(len(clusters))]

print_flag = False
for iteration in range(args.rounds):

    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # idxs_users = comm_users[iteration]

    print(f'###### ROUND {iteration + 1} ######')
    print(f'Clients {idxs_users}')

    """distribution shift"""
    if args.shift_type == 'swap_all':
        swap_data_all(clients, args.swap_p)
    elif args.shift_type == 'rotate':
        rotate_data(clients, args.swap_p)
    elif args.shift_type == 'swap_part':
        swap_data_part(clients, args.swap_p)
    elif args.shift_type == 'incremental' and iteration != 0:
        increase_data(iteration, clients)

    schedule_clients(args, clients, idxs_users, clusters, cluster_dW, clients_clust_id)

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

    # IntraGroupUpdate
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
        # if acc > best_glob_acc[k]:
        #     best_glob_acc[k] = acc

    # InterGroupAggregation
    agg_lr = 0
    w_clusters = copy.deepcopy(w_glob_per_cluster)

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

    # refresh traindata_cls_counts
    traindata_cls_counts = {}
    for client_id in range(args.num_users):
        unq, unq_cnt = np.unique(clients[client_id].ldr_train.dataset.target, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        traindata_cls_counts[client_id] = tmp

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

# for jj in range(len(clusters)):
#     print(f'Cluster {jj}, Best Glob Acc {best_glob_acc[jj]:.3f}')
#
# print(f'Average Best Glob Acc {np.mean(best_glob_acc[0:len(clusters)]):.3f}')
print(f'max final_tacc_pr: {max(final_tacc_pr)}')

print(f'ckp_avg_tacc: {ckp_avg_tacc}')

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')
