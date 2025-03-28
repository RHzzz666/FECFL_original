import sys
import numpy as np

import copy
import os
import gc
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
sys.path.append('../')

from src.dynamic.distribution_shift import *
from src.models import *
from src.fedavg import *
from src.client import *
from src.clustering import *
from src.utils import *
from datasets_models import get_datasets, init_nets
from src.byzantine.byzantine_shift import inject_noise_samples, inject_adversarial_samples, inject_backdoor_hsv, inject_backdoor_pixel_pattern, inject_backdoor_rotation, inject_backdoor_blur, inject_backdoor_inversion, inject_backdoor_crop, inject_backdoor_contrast, inject_fgsm_adversarial, inject_rotation_adversarial, parse_malicious_clients


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

    clients.append(Client_FedAvg(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                                 args.lr, args.momentum, args.device, train_ds_local, test_ds_local))

if args.shift_type == 'incremental':
    increase_data(0, clients)

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

users_best_acc = [0 for _ in range(args.num_users)]
best_glob_acc = 0
benign_avg_acc_per_round = []

w_glob = copy.deepcopy(initial_state_dict)
print_flag = False
for iteration in range(args.rounds):

    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # idxs_users = comm_users[iteration]

    print(f'###### ROUND {iteration + 1} ######')
    print(f'Clients {idxs_users}')

    """distribution shift and byzantine attack"""
    if args.shift_type == 'swap_all':
        swap_data_all(clients, args.swap_p)
    elif args.shift_type == 'rotate':
        rotate_data(clients, args.swap_p)
    elif args.shift_type == 'swap_part':
        swap_data_part(clients, args.swap_p)
    elif args.shift_type == 'incremental' and iteration != 0:
        increase_data(iteration, clients)
    elif args.shift_type == 'noise_injection':
        # Inject noise samples to a fraction of clients
        if iteration == 10:  # Only apply the attack once at the round 10
            # 解析恶意客户端列表或使用随机选择
            malicious_clients = parse_malicious_clients(args)
            if malicious_clients is None:
                num_malicious = max(int(args.swap_p * args.num_users), 1)
                malicious_clients = np.random.choice(range(args.num_users), num_malicious, replace=False)
            else:
                num_malicious = len(malicious_clients)
                print(f"Using specified malicious clients: {malicious_clients}")

            # Get the noise parameters
            noise_ratio = getattr(args, 'noise_ratio', 0.2)
            noise_type = getattr(args, 'noise_type', 'pure')
            noise_level = getattr(args, 'noise_level', 0.5)

            inject_noise_samples(clients, malicious_clients, noise_ratio, noise_type, noise_level)
            print(f"Injected {noise_type} noise samples with ratio {noise_ratio} and level {noise_level} to {num_malicious} clients: {malicious_clients}")

            # Store the malicious clients for later analysis
            args.malicious_clients_list = malicious_clients
    elif args.shift_type == 'backdoor_hsv':
        # Apply backdoor attack with HSV color space transformation as trigger
        if iteration == 10:  # Only apply the attack once at the round 10
            # 解析恶意客户端列表或使用随机选择
            malicious_clients = parse_malicious_clients(args)
            if malicious_clients is None:
                num_malicious = max(int(args.swap_p * args.num_users), 1)
                malicious_clients = np.random.choice(range(args.num_users), num_malicious, replace=False)
            else:
                num_malicious = len(malicious_clients)
                print(f"Using specified malicious clients: {malicious_clients}")

            # Get the backdoor parameters
            trigger_ratio = getattr(args, 'trigger_ratio', 0.2)
            target_class = getattr(args, 'target_class', None)
            random_target = getattr(args, 'random_target', False)

            inject_backdoor_hsv(clients, malicious_clients, target_class, trigger_ratio, random_target)
            target_str = "random labels" if random_target else f"target class {target_class}" if target_class is not None else "original labels"
            print(f"Injected HSV backdoor with ratio {trigger_ratio} and {target_str} to {num_malicious} clients: {malicious_clients}")

            # Store the malicious clients for later analysis
            args.malicious_clients_list = malicious_clients
    elif args.shift_type == 'rotation_adversarial':
        # 应用基于旋转变换的对抗攻击
        if iteration == 10:  # 在第10轮应用攻击
            # 解析恶意客户端列表或使用随机选择
            malicious_clients = parse_malicious_clients(args)
            if malicious_clients is None:
                num_malicious = max(int(args.swap_p * args.num_users), 1)
                malicious_clients = np.random.choice(range(args.num_users), num_malicious, replace=False)
            else:
                num_malicious = len(malicious_clients)
                print(f"使用指定的恶意客户端: {malicious_clients}")

            # 获取攻击参数
            epsilon = getattr(args, 'epsilon', 0.1)
            target_class = getattr(args, 'target_class', None)
            random_target = getattr(args, 'random_target', False)
            attack_ratio = getattr(args, 'attack_ratio', 0.2)
            num_classes = getattr(args, 'num_classes', 10)

            # 使用全局模型作为目标模型
            target_model = copy.deepcopy(net_glob)
            target_model.to(args.device)
            target_model.eval()

            inject_rotation_adversarial(clients, malicious_clients, target_model, epsilon,
                                        target_class, attack_ratio, random_target, args.device, num_classes)
            target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "原始标签"
            print(f"注入旋转对抗样本，epsilon={epsilon}，比例={attack_ratio}，{target_str}，攻击客户端数量={num_malicious}：{malicious_clients}")

            # 存储恶意客户端列表供后续分析
            args.malicious_clients_list = malicious_clients

    for idx in idxs_users:

        clients[idx].set_state_dict(copy.deepcopy(w_glob))

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

    total_data_points = sum([train_datanum_list[r] for r in idxs_users])
    fed_avg_freqs = [train_datanum_list[r] / total_data_points for r in idxs_users]

    w_locals = []
    for idx in idxs_users:
        w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))

    ww = FedAvg(w_locals, weight_avg=fed_avg_freqs)
    w_glob = copy.deepcopy(ww)
    net_glob.load_state_dict(copy.deepcopy(ww))
    _, acc = eval_test(net_glob, args, test_dl_global)
    if acc > best_glob_acc:
        best_glob_acc = acc

        # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    avg_init_tacc = sum(init_local_tacc) / len(init_local_tacc)
    avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)
    avg_final_tacc = sum(final_local_tacc) / len(final_local_tacc)

    print('## END OF ROUND ##')
    template = 'Average Train loss {:.3f}'
    print(template.format(loss_avg))

    #     template = "AVG Init Test Loss: {:.3f}, AVG Init Test Acc: {:.3f}"
    #     print(template.format(avg_init_tloss, avg_init_tacc))

    #     template = "AVG Final Test Loss: {:.3f}, AVG Final Test Acc: {:.3f}"
    #     print(template.format(avg_final_tloss, avg_final_tacc))

    template = "Global Model Test Acc: {:.3f}, Global Model Best Test Acc: {:.3f}"
    print(template.format(acc, best_glob_acc))

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
        malicious_acc = []
        benign_acc = []
        for k in range(args.num_users):
            loss, acc = clients[k].eval_test()
            current_acc.append(acc)

            if acc > clients_best_acc[k]:
                clients_best_acc[k] = acc

            # Track malicious and benign clients separately
            if hasattr(args, 'malicious_clients_list') and k in args.malicious_clients_list:
                malicious_acc.append(acc)
            else:
                benign_acc.append(acc)

            template = ("Client {:3d}, labels {}, count {}, best_acc {:3.3f}, current_acc {:3.3f} \n")
            print(template.format(k, traindata_cls_counts[k], clients[k].get_count(),
                                  clients_best_acc[k], current_acc[-1]))

        template = ("Round {:1d}, Avg current_acc {:3.3f}, Avg best_acc {:3.3f}")
        print(template.format(iteration + 1, np.mean(current_acc), np.mean(clients_best_acc)))

        # Print separate metrics for malicious and benign clients if label flipping was applied
        template = ("Malicious clients avg acc: {:3.3f}, Benign clients avg acc: {:3.3f}")
        print(template.format(np.mean(malicious_acc) if len(malicious_acc) > 0 else 0,
                              np.mean(benign_acc) if len(benign_acc) > 0 else 0))

        benign_avg_acc_per_round.append(np.mean(benign_acc) if len(benign_acc) > 0 else 0)



        ckp_avg_tacc.append(np.mean(current_acc))
        ckp_avg_best_tacc.append(np.mean(clients_best_acc))

    print('----- Analysis End of Round -------')
    for idx in idxs_users:
        print(f'Client {idx}, Count: {clients[idx].get_count()}, Labels: {traindata_cls_counts[idx]}')

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

benign_train_acc = []
benign_test_acc = []
benign_test_loss = []
benign_train_loss = []

for idx in range(args.num_users):
    loss, acc = clients[idx].eval_test()

    test_loss.append(loss)
    test_acc.append(acc)

    loss, acc = clients[idx].eval_train()

    train_loss.append(loss)
    train_acc.append(acc)

    if hasattr(args, 'malicious_clients_list') and idx not in args.malicious_clients_list:
        # Track only benign clients for train and test accuracy
        benign_train_acc.append(acc)
        benign_test_acc.append(acc)
        benign_train_loss.append(loss)
        benign_test_loss.append(loss)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_acc) / len(test_acc)

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_acc) / len(train_acc)

benign_test_acc = sum(benign_test_acc) / len(benign_test_acc)
benign_train_acc = sum(benign_train_acc) / len(benign_train_acc)
benign_test_loss = sum(benign_test_loss) / len(benign_test_loss)
benign_train_loss = sum(benign_train_loss) / len(benign_train_loss)


print(f'Best Clients AVG Acc: {np.mean(clients_best_acc)}')

net_glob.load_state_dict(copy.deepcopy(w_glob))
_, acc = eval_test(net_glob, args, test_dl_global)
if acc > best_glob_acc:
    best_glob_acc = acc

template = "Global Model Test Acc: {:.3f}, Global Model Best Test Acc: {:.3f}"
print(template.format(acc, best_glob_acc))
print(f'max final_tacc_pr: {max(final_tacc_pr)}')
print(f'ckp_avg_tacc: {ckp_avg_tacc}')

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')

print(f'Benign Test Loss: {benign_test_loss}')
print(f'Benign Test Acc: {benign_test_acc}')