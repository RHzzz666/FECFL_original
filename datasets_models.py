import copy


from src.data_process_fedlab.cifar10.rotated_cifar10 import RotatedCIFAR10
from src.data_process_fedlab.cifar10.original_cifar10 import OriginalCIFAR10
from src.data_process_fedlab.cifar10.rgb_hsv_cifar10 import RGBHSVCIAFR10
from src.data_process_fedlab.cifar10.mix_cifar10 import MixCIFAR10
from src.data_process_fedlab.fmnist.original_fmnist import OriginalFMNIST
from src.data_process_fedlab.fmnist.rotated_fmnist import RotatedFMNIST
from src.data_process_fedlab.fmnist.mix_fmnist import MixFMNIST
from src.data_process_fedlab.cinic10.original_cinic10 import OriginalCINIC10
from src.data_process_fedlab.cinic10.rotated_cinic10 import RotatedCINIC10
from src.data_process_fedlab.cinic10.rgb_hsv_cinic10 import RGBHSVCINIC10
from src.data_process_fedlab.cinic10.mix_cinic10 import MixCINIC10
from src.data_process_fedlab.cifar100.original_cifar100 import OriginalCIFAR100
from src.data_process_fedlab.cifar100.rotated_cifar100 import RotatedCIFAR100
from src.data_process_fedlab.cifar100.rgb_hsv_cifar100 import RGBHSVCIFAR100
from src.data_process_fedlab.cifar100.mix_cifar100 import MixCIFAR100
from src.data_process_fedlab.stl10.original_stl10 import OriginalSTL10
from src.data_process_fedlab.tiny.original_tiny import OriginalTINY
from src.data_process_fedlab.tiny.rotated_tiny import RotatedTINY
from src.data_process_fedlab.tiny.rgb_hsv_tiny import RGBHSVTINY
from src.utils import *
from src.models import *
from src.models.autoencoders import ConvAE

args = args_parser()

def get_datasets(args):
    if args.dataset == "cifar10" and args.partition == "rotated":
        print("dataset = cifar10, partition_method = rotated")
        data_dir = '../data/cifar10'
        save_dir = '../data/cifar10_rotated'
        data_set = RotatedCIFAR10(data_dir, save_dir, args.num_users)
    elif args.dataset == "cifar10" and args.partition == "pathological":
        print("dataset = cifar10, partition_method = pathological")
        data_dir = '../data/cifar10'
        save_dir = '../data/cifar10_pathological'
        data_set = OriginalCIFAR10(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "cifar10" and args.partition == "rgb_hsv":
        print("dataset = cifar10, partition_method = rgb_hsv")
        data_dir = '../data/cifar10'
        save_dir = '../data/cifar10_rgb_hsv'
        data_set = RGBHSVCIAFR10(data_dir, save_dir, args.num_users)
    elif args.dataset == "cifar10" and args.partition == "mix":
        print("dataset = cifar10, partition_method = mix")
        data_dir = '../data/cifar10'
        save_dir = '../data/cifar10_mix'
        data_set = MixCIFAR10(data_dir, save_dir, args.num_users)
    elif args.dataset == "cifar10" and args.partition == "noniid-#label2":
        print("dataset = cifar10, partition_method = noniid-#label2")
        data_dir = '../data/cifar10'
        save_dir = '../data/cifar10_noniid-#label2'
        data_set = OriginalCIFAR10(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "fmnist" and args.partition == "pathological":
        print("dataset = fmnist, partition_method = pathological")
        data_dir = '../data/fmnist'
        save_dir = '../data/fmnist_pathological'
        data_set = OriginalFMNIST(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "fmnist" and args.partition == "noniid-#label2":
        print("dataset = fmnist, partition_method = noniid-#label2")
        data_dir = '../data/fmnist'
        save_dir = '../data/fmnist_noniid-#label2'
        data_set = OriginalFMNIST(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "fmnist" and args.partition == "rotated":
        print("dataset = fmnist, partition_method = rotated")
        data_dir = '../data/fmnist'
        save_dir = '../data/fmnist_rotated'
        data_set = RotatedFMNIST(data_dir, save_dir, args.num_users)
    elif args.dataset == "fmnist" and args.partition == "mix":
        print("dataset = fmnist, partition_method = mix")
        data_dir = '../data/fmnist'
        save_dir = '../data/fmnist_mix'
        data_set = MixFMNIST(data_dir, save_dir, args.num_users)
    elif args.dataset == "cinic10" and args.partition == "pathological":
        print("dataset = cinic10, partition_method = pathological")
        data_dir = '../data/cinic10'
        save_dir = '../data/cinic10_pathological'
        data_set = OriginalCINIC10(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "cinic10" and args.partition == "rotated":
        print("dataset = cinic10, partition_method = rotated")
        data_dir = '../data/cinic10'
        save_dir = '../data/cinic10_rotated'
        data_set = RotatedCINIC10(data_dir, save_dir, args.num_users)
    elif args.dataset == "cinic10" and args.partition == "rgb_hsv":
        print("dataset = cinic10, partition_method = rgb_hsv")
        data_dir = '../data/cinic10'
        save_dir = '../data/cinic10_rgb_hsv'
        data_set = RGBHSVCINIC10(data_dir, save_dir, args.num_users)
    elif args.dataset == "cinic10" and args.partition == "mix":
        print("dataset = cinic10, partition_method = mix")
        data_dir = '../data/cinic10'
        save_dir = '../data/cinic10_mix'
        data_set = MixCINIC10(data_dir, save_dir, args.num_users)
    elif args.dataset == "cinic10" and args.partition == "noniid-#label2":
        print("dataset = cinic10, partition_method = noniid-#label2")
        data_dir = '../data/cinic10'
        save_dir = '../data/cinic10_noniid-#label2'
        data_set = OriginalCINIC10(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "cifar100" and args.partition == "pathological":
        print("dataset = cifar100, partition_method = pathological")
        data_dir = '../data/cifar100'
        save_dir = '../data/cifar100_pathological'
        data_set = OriginalCIFAR100(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "cifar100" and args.partition == "rotated":
        print("dataset = cifar100, partition_method = rotated")
        data_dir = '../data/cifar100'
        save_dir = '../data/cifar100_rotated'
        data_set = RotatedCIFAR100(data_dir, save_dir, args.num_users)
    elif args.dataset == "cifar100" and args.partition == "rgb_hsv":
        print("dataset = cifar100, partition_method = rgb_hsv")
        data_dir = '../data/cifar100'
        save_dir = '../data/cifar100_rgb_hsv'
        data_set = RGBHSVCIFAR100(data_dir, save_dir, args.num_users)
    elif args.dataset == "cifar100" and args.partition == "mix":
        print("dataset = cifar100, partition_method = mix")
        data_dir = '../data/cifar100'
        save_dir = '../data/cifar100_mix'
        data_set = MixCIFAR100(data_dir, save_dir, args.num_users)
    elif args.dataset == "cifar100" and args.partition == "noniid-#label20":
        print("dataset = cifar100, partition_method = noniid-#label20")
        data_dir = '../data/cifar100'
        save_dir = '../data/cifar100_noniid-#label20'
        data_set = OriginalCIFAR100(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "stl10" and args.partition == "pathological#2":
        print("dataset = stl10, partition_method = pathological#2")
        data_dir = '../data/stl10'
        save_dir = '../data/stl10_pathological#2'
        data_set = OriginalSTL10(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "tiny" and args.partition == "pathological":
        print("dataset = tiny, partition_method = pathological")
        data_dir = '../data/tiny-imagenet-200'
        save_dir = '../data/tiny_pathological'
        data_set = OriginalTINY(data_dir, save_dir, args.num_users, args.partition)
    elif args.dataset == "tiny" and args.partition == "rotated":
        print("dataset = tiny, partition_method = rotated")
        data_dir = '../data/tiny-imagenet-200'
        save_dir = '../data/tiny_rotated'
        data_set = RotatedTINY(data_dir, save_dir, args.num_users)
    elif args.dataset == "tiny" and args.partition == "rgb_hsv":
        print("dataset = tiny, partition_method = rgb_hsv")
        data_dir = '../data/tiny-imagenet-200'
        save_dir = '../data/tiny_rgb_hsv'
        data_set = RGBHSVTINY(data_dir, save_dir, args.num_users)
    else:
        raise NotImplementedError
    return data_set


################################### build model
def init_nets(args, dropout_p=0.5):
    users_model = []

    for net_i in range(-1, args.num_users):
        if args.dataset == "generated":
            net = PerceptronModel().to(args.device)
        elif args.model == "simple-mlp":
            net = SimpleMLP().to(args.device)
        elif args.model == "simple-cnn":  # Lenet-5
            if args.dataset in ("cifar10", "cinic10"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
        elif args.model == 'resnet9':
            if args.dataset == 'cifar100':
                net = ResNet9(in_channels=3, num_classes=100)
            elif args.dataset == 'stl10':
                net = ResNet9(in_channels=3, num_classes=100, dim=4608)
            elif args.dataset == 'tiny':
                net = ResNet9(in_channels=3, num_classes=200, dim=512 * 2 * 2)
        elif args.unsupervised:
            net = ConvAE().to(args.device)
        else:
            print("not supported yet")
            exit(1)
        if net_i == -1:
            net_glob = copy.deepcopy(net)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            server_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                server_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            users_model[net_i].load_state_dict(initial_state_dict)

    return users_model, net_glob, initial_state_dict, server_state_dict


data_set = get_datasets(args)

if args.partition == "rotated":
    data_set.preprocess(thetas=[0, 90, 180, 270])
else:
    data_set.preprocess()


