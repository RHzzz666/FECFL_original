import random
import numpy as np
import os
import gc
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms

from src.fedavg import *
from src.client.client_fecfl import Client_FECFL
from src.clustering import *
from src.utils import *
from datasets_models import get_datasets, init_nets

args = args_parser()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu)  ## Setting cuda on GPU
print('Using GPU: {} '.format(torch.cuda.current_device()))
print('Using Device: {} '.format(args.device))
# Reproducability
# ----------------------------------
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)


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

################################### new data partitioning section

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
features_list = []

for idx in range(args.num_users):
    print(f'Client {idx}, Labels: {traindata_cls_counts[idx]}')

for idx in range(args.num_users):
    train_ds_local = train_ds_list[idx]
    test_ds_local = test_ds_list[idx]

    clients.append(Client_FECFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                                args.lr, args.momentum, args.device, train_ds_local, test_ds_local))

    features = clients[idx].extract_features_avg()
    # features = clients[idx].extract_features_norm()
    features_list.append(features)


###################################### Clustering
np.set_printoptions(precision=2)
# m = max(int(args.frac * args.num_users), 1)
# clients_idxs = np.random.choice(range(args.num_users), m, replace=False)

cnt = args.num_users
for r in range(1):
    print(f'Round {r}')
    clients_idxs = np.arange(cnt)

    sim_matrix = []
    for i, feature1 in enumerate(features_list):
        row = []
        for j, feature2 in enumerate(features_list):
            # cos
            sim = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
            if i == j:
                sim = 1.0
            row.append(sim)
        sim_matrix.append(row)
    sim_matrix = np.array(sim_matrix)

    distance_matrix = 1 - sim_matrix

    # ### EMD
    # n = len(features_list)
    # emd_matrix = np.zeros((n, n))  # 初始化EMD距离矩阵
    #
    # for i in range(n):
    #     for j in range(i + 1, n):  # 只计算矩阵的上半部分，因为它是对称的
    #         vec1 = features_list[i]
    #         vec2 = features_list[j]
    #
    #         # 假设每个点的重要性相同，使用均匀分布
    #         weights1 = np.ones((vec1.shape[0],)) / vec1.shape[0]
    #         weights2 = np.ones((vec2.shape[0],)) / vec2.shape[0]
    #
    #         # 计算成本矩阵
    #         cost_matrix = ot.dist(vec1, vec2, metric='cosine')
    #
    #         # 计算EMD距离
    #         emd_distance = ot.emd2(weights1, weights2, cost_matrix)
    #
    #         # 填充EMD矩阵
    #         emd_matrix[i, j] = emd_distance
    #         emd_matrix[j, i] = emd_distance  # EMD距离是对称的
    # distance_matrix = emd_matrix

    print('')
    print('Distance Matrix')
    print(distance_matrix.tolist())

    print('')
    print("Cluster threshold")
    print(args.cluster_alpha)
    # hc
    if args.cluster_algo == 'hc':  # default
        clusters = hierarchical_clustering(copy.deepcopy(np.array(distance_matrix)), thresh=args.cluster_alpha, linkage=args.linkage)
    elif args.cluster_algo == 'dbscan':
        cluster_algo = DBSCAN(min_samples=2, metric='precomputed', eps=args.cluster_alpha)
        cluster_labels = cluster_algo.fit_predict(distance_matrix.tolist())

        unique_labels = set(cluster_labels)
        clusters = [list(np.where(cluster_labels == label)[0]) for label in unique_labels]
    elif args.cluster_algo == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=args.nclusters, max_iter=20, random_state=args.seed).fit(distance_matrix)
        cluster_labels = kmeans.labels_

        unique_labels = set(cluster_labels)
        clusters = [list(np.where(cluster_labels == label)[0]) for label in unique_labels]

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

#### TSNE plot
# ground_truth = [[i for i in range(0, 20)], [i for i in range(20, 40)], [i for i in range(40, 60)],
#                 [i for i in range(60, 80)], [i for i in range(80, 100)]]
#
# tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
# tsne_results = tsne.fit_transform(np.array(features_list))
#
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  '#8c564b',  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#
# labels = ['Label: {0, 1}', 'Label: {2, 3}', 'Label: {4, 5}', 'Label: {6, 7}', 'Label: {8, 9}']
#
# plt.figure(figsize=(8, 6))
#
# for i, cluster in enumerate(ground_truth):
#     if i >= len(colors):
#         print("Warning: Not enough colors for all clusters, some clusters will have the same color.")
#         break
#
#     cluster_points = tsne_results[[index for index in cluster], :]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=labels[i])
#
# plt.legend(fontsize='large')
# plt.savefig('../tsne_pathological_fecfl.png')

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
