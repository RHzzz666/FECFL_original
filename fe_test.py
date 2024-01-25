import numpy as np

import copy
import os
import gc
import pickle

import torch
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms

from src.models import *
from src.client.client_fecfl import Client_FECFL
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
features_list = []

for idx in range(args.num_users):
    print(f'Client {idx}, Labels: {traindata_cls_counts[idx]}')

for idx in range(args.num_users):
    train_ds_local = train_ds_list[idx]
    test_ds_local = test_ds_list[idx]

    clients.append(Client_FECFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                                args.lr, args.momentum, args.device, train_ds_local, test_ds_local))

    features = clients[idx].extract_features_avg()
    features_list.append(features)

loss_locals = []

for i in range(100):
    avg_loss = 0.0
    ### TSNE plot
    ground_truth = [[i for i in range(0, 25)], [i for i in range(25, 50)], [i for i in range(50, 75)], [i for i in range(75, 100)]]

    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(np.array(features_list))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    labels = ['Rotate 0', 'Rotate 90', 'Rotate 180', 'Rotate 270']

    plt.figure(figsize=(8, 6))

    for j, cluster in enumerate(ground_truth):
        if j >= len(colors):
            print("Warning: Not enough colors for all clusters, some clusters will have the same color.")
            break

        cluster_points = tsne_results[[index for index in cluster], :]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[j], label=labels[j])

    plt.legend(fontsize=15)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('../saved_images/tsne_rotated_fecfl_' + str(i) + '.png')

    features_list = []
    for idx in range(args.num_users):
        loss = clients[idx].train(is_print=False)
        avg_loss += loss

        features = clients[idx].extract_features_avg()
        features_list.append(features)
    avg_loss /= args.num_users
    loss_locals.append(copy.deepcopy(avg_loss))

plt.figure(figsize=(8, 6))
plt.plot(range(len(loss_locals)), loss_locals)
plt.ylabel('train_loss')
plt.xlabel('round')
plt.savefig('../saved_images/loss_rotated_fecfl.png')



