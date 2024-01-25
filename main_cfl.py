import numpy as np

import copy
import os
import gc

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms

from src.models import *
from src.fedavg import *
from src.client.client_cfl import *
from src.utils import *
from datasets_models import data_set, init_nets

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
    
path = args.savedir + args.alg + '/' + args.partition + '/' + args.dataset + '/' #+ str(args.trial)
mkdirs(path)

template = "Algorithm {}, Clients {}, Dataset {}, Model {}, Non-IID {}, Threshold {}, K {}, Linkage {}, LR {}, Ep {}, Rounds {}, bs {}, frac {}"

s = template.format(args.alg, args.num_users, args.dataset, args.model, args.partition, args.cluster_alpha, args.n_basis, args.linkage, args.lr, args.local_ep, args.rounds, args.local_bs, args.frac)

print(s)

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
    #print(np.array(param.data.cpu().numpy().reshape([-1])))
    #print(isinstance(param.data.cpu().numpy(), np.array))
print(total)

    
################################# Initializing Clients 

m = max(int(args.frac * args.num_users), 1)
clients = []

for idx in range(args.num_users):
    train_ds_local = train_ds_list[idx]
    test_ds_local = test_ds_list[idx]
    train_dl_local = DataLoader(train_ds_local, batch_size=args.local_bs, shuffle=True, drop_last=True)
    test_dl_local = DataLoader(test_ds_local, batch_size=args.local_bs, shuffle=False)
    
    clients.append(Client_CFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, train_dl_local, test_dl_local))
    
EPS_1 = 0.4
EPS_2 = 1.6
    
###################################### Federation 

float_formatter = "{:.4f}".format
#np.set_printoptions(formatter={float: float_formatting_function})
np.set_printoptions(formatter={'float_kind':float_formatter})

loss_train = []

init_tracc_pr = []  # initial train accuracy for each round 
final_tracc_pr = [] # final train accuracy for each round 

init_tacc_pr = []  # initial test accuarcy for each round 
final_tacc_pr = [] # final test accuracy for each round

init_tloss_pr = []  # initial test loss for each round 
final_tloss_pr = [] # final test loss for each round 

clients_best_acc = [0 for _ in range(args.num_users)]
w_locals, loss_locals = [], []

init_local_tacc = []       # initial local test accuracy at each round 
final_local_tacc = []  # final local test accuracy at each round 

init_local_tloss = []      # initial local test loss at each round 
final_local_tloss = []     # final local test loss at each round 

ckp_avg_tacc = []
ckp_avg_best_tacc = []


users_best_acc = [0 for _ in range(args.num_users)]

cluster_indices = [np.arange(args.num_users).astype("int")]
client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

print_flag = True
for iteration in range(args.rounds):

    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    
    print(f'###### ROUND {iteration+1} ######')
    print(f'Clients {idxs_users}')
    loss_locals = []

    for idx in idxs_users:
        loss, acc = clients[idx].eval_test()        

        init_local_tacc.append(acc)
        init_local_tloss.append(loss)

        loss = clients[idx].train(is_print=False)
        loss_locals.append(copy.deepcopy(loss))

    similarities = compute_pairwise_similarities(clients)

    cluster_indices_new = []
    for idc in cluster_indices:
        max_norm = compute_max_update_norm([clients[i] for i in idc])
        mean_norm = compute_mean_update_norm([clients[i] for i in idc])
             
        if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc)>2 and iteration>20:
            print("bipartitioning")
            c1, c2 = cluster_clients(similarities[idc][:,idc]) 
            cluster_indices_new += [c1, c2]
        else:
            cluster_indices_new += [idc]

    cluster_indices = cluster_indices_new
    print(f'Cluster indices: {cluster_indices}')
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    aggregate_clusterwise(client_clusters)

    for idx in idxs_users:
        loss, acc = clients[idx].eval_test()

        if acc > clients_best_acc[idx]:
            clients_best_acc[idx] = acc

        final_local_tacc.append(acc)
        final_local_tloss.append(loss)

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
        print(template.format(iteration+1, np.mean(current_acc), np.mean(clients_best_acc)))
        
        ckp_avg_tacc.append(np.mean(current_acc))
        ckp_avg_best_tacc.append(np.mean(clients_best_acc))
    
    print('----- Analysis End of Round -------')
    for idx in idxs_users:
        print(f'Client {idx}, Count: {clients[idx].get_count()}, Labels: {traindata_cls_counts[idx]}')

    print('')
    
    loss_train.append(loss_avg)
    
    init_tacc_pr.append(avg_init_tacc)
    init_tloss_pr.append(avg_init_tloss)
    
    final_tacc_pr.append(avg_final_tacc)
    final_tloss_pr.append(avg_final_tloss)
    
    #break;
    ## clear the placeholders for the next round 
    w_locals.clear()
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
print(f'max final_tacc_pr: {max(final_tacc_pr)}')
print(f'ckp_avg_tacc: {ckp_avg_tacc}')


print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')