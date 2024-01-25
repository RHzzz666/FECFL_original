import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--nclass', type=int, default=2, help="classes or shards per user")
    parser.add_argument('--nsample_pc', type=int, default=250, 
                        help="number of samples per class or shard for each client")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--warmup_epoch', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--trial', type=int, default=1, help="the trial number")
    parser.add_argument('--mu', type=float, default=0.001, help="FedProx Regularizer")


    # model arguments
    parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--ks', type=int, default=5, help='kernel size to use for convolutions')
    parser.add_argument('--in_ch', type=int, default=3, help='input channels of the first conv layer')

    # dataset partitioning arguments
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help="name of dataset: mnist, cifar10, cifar100")
    parser.add_argument('--noniid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--shard', action='store_true', help='whether non-i.i.d based on shard or not')
    parser.add_argument('--label', action='store_true', help='whether non-i.i.d based on label or not')
    parser.add_argument('--split_test', action='store_true', 
                        help='whether split test set in partitioning or not')
    
    # NIID Benchmark dataset partitioning 
    parser.add_argument('--savedir', type=str, default='../save/', help='save directory')
    parser.add_argument('--datadir', type=str, default='../data/', help='data directory')
    parser.add_argument('--logdir', type=str, default='../logs/', help='logs directory')
    parser.add_argument('--partition', type=str, default='noniid-#label2', help='method of partitioning')
    parser.add_argument('--alg', type=str, default='cluster_fl', help='Algorithm')

    # clustering arguments 
    parser.add_argument('--cluster_alpha', type=float, default=3.5, help="the clustering threshold")
    parser.add_argument('--n_basis', type=int, default=5, help="number of basis per label")
    parser.add_argument('--linkage', type=str, default='average', help="Type of Linkage for HC")

    parser.add_argument('--nclasses', type=int, default=10, help="number of classes")
    parser.add_argument('--nsamples_shared', type=int, default=2500, help="number of shared data samples")
    parser.add_argument('--nclusters', type=int, default=5, help="Number of Clusters for IFCA")
    parser.add_argument('--num_incluster_layers', type=int, default=2, help="Number of Clusters for IFCA")
    
    # pruning arguments 
    parser.add_argument('--pruning_percent', type=float, default=10, 
                        help="Pruning percent for layers (0-100)")
    parser.add_argument('--pruning_target', type=int, default=30, 
                        help="Total Pruning target percentage (0-100)")
    parser.add_argument('--dist_thresh', type=float, default=0.0001, 
                        help="threshold for fcs masks difference ")
    parser.add_argument('--acc_thresh', type=int, default=50, 
                        help="accuracy threshold to apply the derived pruning mask")
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    # other arguments 
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--is_print', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=10, help="printing frequency during training rounds")
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1)')

    # unsupervised
    parser.add_argument('--unsupervised', type=bool, default=False, help='whether unsupervised or not')

    # distribution shift
    parser.add_argument('--shift_type', type=str, default='all', help='all, part, incremental')
    parser.add_argument('--swap_p', type=float, default=0.05, help='percent of distribution shift')

    # StoCFL
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='number of groups')

    # FlexCFL # depreciated
    parser.add_argument('--pretrain_epoch', type=int, default=10, help='number of pretrain epoch')

    # FedSoft
    parser.add_argument('--estimation_interval', type=int, default=2, help='interval of importance estimation')

    # group test
    parser.add_argument('--group_test', type=int, default=0, help='whether group test or not')

    # number of classes
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

    # cluster algo
    parser.add_argument('--cluster_algo', type=str, default='hc', help='cluster algorithm, hc or dbscan')

    args = parser.parse_args()
    return args
