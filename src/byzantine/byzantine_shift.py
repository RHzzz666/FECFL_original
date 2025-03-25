import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class LabelFlippingDataset(Dataset):
    def __init__(self, original_dataset, flip_mapping=None):
        """
        A wrapper dataset that flips labels according to a specified mapping.

        Args:
            original_dataset: The original dataset
            flip_mapping: Dictionary mapping from original label to flipped label
                          e.g., {0: 2, 1: 9} would map label 0 to 2 and label 1 to 9
        """
        self.original_dataset = original_dataset
        self.flip_mapping = flip_mapping or {}
        self.target = []

        # Handle the target attribute for compatibility
        if hasattr(original_dataset, 'target'):
            self.target = [self.flip_mapping.get(t, t) for t in original_dataset.target]
        elif hasattr(original_dataset, 'targets'):
            self.target = [self.flip_mapping.get(t, t) for t in original_dataset.targets]
        else:
            for _, t in original_dataset:
                self.target.append(self.flip_mapping.get(t, t))

        # If original dataset has data attribute, replicate it
        if hasattr(original_dataset, 'data'):
            self.data = original_dataset.data

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        # Flip the label if it's in the mapping
        flipped_y = self.flip_mapping.get(y, y)
        return x, flipped_y


def flip_labels(clients, attack_indices, flip_mapping=None, attack_type='default'):
    """
    Apply label flipping attack to selected clients.

    Args:
        clients: List of all clients
        attack_indices: Indices of clients to attack
        flip_mapping: Dictionary mapping from original label to flipped label
                      If None, will use a default mapping based on the dataset
        attack_type: Type of attack to perform:
                     - 'default': Use the provided flip_mapping
                     - 'random': Randomly flip labels
                     - 'targeted': Flip only specific labels that client has most of
                     - 'opposite': Flip to semantically opposite class (needs flip_mapping)
                     - 'adversarial': Flip to the class that would cause most damage (e.g., similar classes)
    """
    # Different types of attack mappings
    if attack_type == 'default' and flip_mapping is None:
        # Default mappings: semantic opposites for CIFAR10
        flip_mapping = {
            0: 2,  # airplane -> bird
            1: 9,  # automobile -> truck
            3: 5,  # cat -> dog
            5: 3,  # dog -> cat
            2: 0,  # bird -> airplane
            9: 1   # truck -> automobile
        }

    for idx in attack_indices:
        print(f"Applying label flipping attack to client {idx}")

        # Get unique labels from the client's dataset
        if hasattr(clients[idx].ds_train, 'target'):
            labels = clients[idx].ds_train.target
        else:
            labels = [y for _, y in clients[idx].ds_train]

        unique_labels = np.unique(labels)

        # Create a mapping based on the attack type
        client_mapping = {}

        if attack_type == 'random':
            # Randomly flip labels to any other class
            all_classes = set(range(10))  # Assuming 10 classes like CIFAR10
            for label in unique_labels:
                other_classes = list(all_classes - {label})
                client_mapping[label] = np.random.choice(other_classes)

        elif attack_type == 'targeted':
            # Find the most common label in this client's dataset
            label_counts = {label: np.sum(np.array(labels) == label) for label in unique_labels}
            most_common_label = max(label_counts, key=label_counts.get)

            # Find a semantically different class to flip to (use the default mapping as a guide)
            default_mapping = {
                0: 2, 1: 9, 3: 5, 5: 3, 2: 0, 9: 1,
                4: 7, 6: 8, 7: 4, 8: 6  # Additional mappings for all CIFAR10 classes
            }

            if most_common_label in default_mapping:
                client_mapping[most_common_label] = default_mapping[most_common_label]

        elif attack_type == 'opposite':
            # Use the provided flip_mapping for semantic opposites
            client_mapping = {k: v for k, v in flip_mapping.items() if k in unique_labels}

        elif attack_type == 'adversarial':
            # Flip to the most similar class to cause subtle but harmful changes
            similarity_mapping = {
                0: 8,  # airplane -> ship (both transportation)
                1: 9,  # automobile -> truck (both vehicles)
                2: 0,  # bird -> airplane (both fly)
                3: 5,  # cat -> dog (both pets)
                4: 7,  # deer -> horse (both animals)
                5: 3,  # dog -> cat (both pets)
                6: 4,  # frog -> deer (both animals)
                7: 1,  # horse -> automobile (transportation confusion)
                8: 0,  # ship -> airplane (both transportation)
                9: 1   # truck -> automobile (both vehicles)
            }
            client_mapping = {k: v for k, v in similarity_mapping.items() if k in unique_labels}

        else:
            # Default case: use the provided mapping
            client_mapping = {k: v for k, v in flip_mapping.items() if k in unique_labels}

        print(f"Client {idx} label flipping map: {client_mapping}")

        # Apply the mapping
        flipped_train_ds = LabelFlippingDataset(clients[idx].ds_train, client_mapping)
        flipped_test_ds = LabelFlippingDataset(clients[idx].ds_test, client_mapping)

        clients[idx].ds_train = flipped_train_ds
        clients[idx].ds_test = flipped_test_ds
        clients[idx].refresh_dl()


class PartialLabelFlippingDataset(Dataset):
    def __init__(self, original_dataset, flip_mapping=None, flip_ratio=0.5):
        """
        A wrapper dataset that flips only a percentage of labels according to a specified mapping.

        Args:
            original_dataset: The original dataset
            flip_mapping: Dictionary mapping from original label to flipped label
            flip_ratio: Float between 0 and 1 indicating what portion of samples to flip
        """
        self.original_dataset = original_dataset
        self.flip_mapping = flip_mapping or {}
        self.flip_ratio = flip_ratio
        self.target = []

        # Store original labels
        if hasattr(original_dataset, 'target'):
            orig_labels = original_dataset.target
        elif hasattr(original_dataset, 'targets'):
            orig_labels = original_dataset.targets
        else:
            orig_labels = [t for _, t in original_dataset]

        # For each class, determine which samples to flip
        self.flip_indices = {}
        for label in set(orig_labels):
            if label in self.flip_mapping:
                indices = [i for i, l in enumerate(orig_labels) if l == label]
                num_to_flip = int(len(indices) * self.flip_ratio)
                np.random.shuffle(indices)
                self.flip_indices[label] = set(indices[:num_to_flip])

        # Generate the target list
        for i, label in enumerate(orig_labels):
            if label in self.flip_mapping and i in self.flip_indices.get(label, set()):
                self.target.append(self.flip_mapping[label])
            else:
                self.target.append(label)

        # If original dataset has data attribute, replicate it
        if hasattr(original_dataset, 'data'):
            self.data = original_dataset.data

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        # Flip the label if it's in the mapping and idx is in the flip set
        if y in self.flip_mapping and idx in self.flip_indices.get(y, set()):
            return x, self.flip_mapping[y]
        return x, y


def partial_flip_labels(clients, attack_indices, flip_mapping=None, attack_type='default', flip_ratio=0.5):
    """
    Apply partial label flipping attack to selected clients.
    Similar to flip_labels, but only flips a percentage of samples for each class.

    Args:
        clients: List of all clients
        attack_indices: Indices of clients to attack
        flip_mapping: Dictionary mapping from original label to flipped label
        attack_type: Type of attack (see flip_labels for options)
        flip_ratio: Percentage of samples to flip for each class (0.0-1.0)
    """
    # Create the flip mapping the same way as in flip_labels
    if attack_type == 'default' and flip_mapping is None:
        flip_mapping = {
            0: 2,  # airplane -> bird
            1: 9,  # automobile -> truck
            3: 5,  # cat -> dog
            5: 3,  # dog -> cat
            2: 0,  # bird -> airplane
            9: 1   # truck -> automobile
        }

    for idx in attack_indices:
        print(f"Applying partial label flipping attack to client {idx}, flip ratio: {flip_ratio}")

        # Get unique labels from the client's dataset
        if hasattr(clients[idx].ds_train, 'target'):
            labels = clients[idx].ds_train.target
        else:
            labels = [y for _, y in clients[idx].ds_train]

        unique_labels = np.unique(labels)

        # Create a mapping based on the attack type (same as in flip_labels)
        client_mapping = {}

        if attack_type == 'random':
            all_classes = set(range(10))  # Assuming 10 classes like CIFAR10
            for label in unique_labels:
                other_classes = list(all_classes - {label})
                client_mapping[label] = np.random.choice(other_classes)

        elif attack_type == 'targeted':
            label_counts = {label: np.sum(np.array(labels) == label) for label in unique_labels}
            most_common_label = max(label_counts, key=label_counts.get)

            default_mapping = {
                0: 2, 1: 9, 3: 5, 5: 3, 2: 0, 9: 1,
                4: 7, 6: 8, 7: 4, 8: 6
            }

            if most_common_label in default_mapping:
                client_mapping[most_common_label] = default_mapping[most_common_label]

        elif attack_type == 'opposite':
            client_mapping = {k: v for k, v in flip_mapping.items() if k in unique_labels}

        elif attack_type == 'adversarial':
            similarity_mapping = {
                0: 8,  # airplane -> ship
                1: 9,  # automobile -> truck
                2: 0,  # bird -> airplane
                3: 5,  # cat -> dog
                4: 7,  # deer -> horse
                5: 3,  # dog -> cat
                6: 4,  # frog -> deer
                7: 1,  # horse -> automobile
                8: 0,  # ship -> airplane
                9: 1   # truck -> automobile
            }
            client_mapping = {k: v for k, v in similarity_mapping.items() if k in unique_labels}

        else:
            client_mapping = {k: v for k, v in flip_mapping.items() if k in unique_labels}

        print(f"Client {idx} partial label flipping map: {client_mapping}, ratio: {flip_ratio}")

        # Apply the partial mapping
        flipped_train_ds = PartialLabelFlippingDataset(clients[idx].ds_train, client_mapping, flip_ratio)
        flipped_test_ds = PartialLabelFlippingDataset(clients[idx].ds_test, client_mapping, flip_ratio)

        clients[idx].ds_train = flipped_train_ds
        clients[idx].ds_test = flipped_test_ds
        clients[idx].refresh_dl()

class NoiseInjectionDataset(Dataset):
    def __init__(self, original_dataset, noise_level=0.1):
        """
        A wrapper dataset that adds Gaussian noise to the data samples.

        Args:
            original_dataset: The original dataset
            noise_level: Standard deviation of the Gaussian noise to be added
        """
        self.original_dataset = original_dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        noise = torch.randn_like(x) * self.noise_level
        noisy_x = x + noise
        return noisy_x, y


class NoiseDataset(Dataset):
    def __init__(self, original_dataset, noise_ratio=0.2, noise_type='gaussian', noise_level=0.5):
        """
        给数据集注入噪声图像
        
        Args:
            original_dataset: 原始数据集
            noise_ratio: 注入噪声数据的比例
            noise_type: 噪声类型 ('gaussian', 'uniform', 'salt_pepper', 'pure')
            noise_level: 噪声强度 (0.0-1.0)
        """
        self.original_dataset = original_dataset
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.target = []
        
        # 保存原始标签
        if hasattr(original_dataset, 'target'):
            self.target = original_dataset.target.copy() if hasattr(original_dataset.target, 'copy') else original_dataset.target
        elif hasattr(original_dataset, 'targets'):
            self.target = original_dataset.targets.copy() if hasattr(original_dataset.targets, 'copy') else original_dataset.targets
        else:
            for _, t in original_dataset:
                self.target.append(t)
                
        # 如果原始数据集有data属性，复制它
        if hasattr(original_dataset, 'data'):
            self.data = original_dataset.data
            
        # 确定要注入噪声的样本索引
        self.total_samples = len(original_dataset)
        self.noise_samples = int(self.total_samples * noise_ratio)
        self.noise_indices = set(np.random.choice(range(self.total_samples), 
                                               self.noise_samples, replace=False))

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        # 如果当前索引在噪声样本集中，注入噪声
        if idx in self.noise_indices:
            # 获取张量形状
            if isinstance(x, np.ndarray):
                shape = x.shape
                dtype = x.dtype
            else:  # 假设是PyTorch张量
                shape = x.shape
                dtype = x.dtype
                
            # 根据噪声类型生成噪声
            if self.noise_type == 'gaussian':
                if isinstance(x, np.ndarray):
                    noise = np.random.normal(0, self.noise_level, shape).astype(dtype)
                    x = x + noise
                    x = np.clip(x, 0, 1) if x.max() <= 1 else np.clip(x, 0, 255)
                else:
                    noise = torch.randn_like(x) * self.noise_level
                    x = x + noise
                    x = torch.clamp(x, 0, 1) if x.max() <= 1 else torch.clamp(x, 0, 255)
                    
            elif self.noise_type == 'uniform':
                if isinstance(x, np.ndarray):
                    noise = np.random.uniform(-self.noise_level, self.noise_level, shape).astype(dtype)
                    x = x + noise
                    x = np.clip(x, 0, 1) if x.max() <= 1 else np.clip(x, 0, 255)
                else:
                    noise = (torch.rand_like(x) * 2 - 1) * self.noise_level
                    x = x + noise
                    x = torch.clamp(x, 0, 1) if x.max() <= 1 else torch.clamp(x, 0, 255)
                    
            elif self.noise_type == 'salt_pepper':
                if isinstance(x, np.ndarray):
                    mask = np.random.random(shape) < self.noise_level/2
                    x = np.copy(x)
                    x[mask] = 1 if x.max() <= 1 else 255
                    mask = np.random.random(shape) < self.noise_level/2
                    x[mask] = 0
                else:
                    mask = torch.rand_like(x) < self.noise_level/2
                    x = x.clone()
                    x[mask] = 1 if x.max() <= 1 else 255
                    mask = torch.rand_like(x) < self.noise_level/2
                    x[mask] = 0
                    
            elif self.noise_type == 'pure':
                # 生成纯随机噪声图像，完全替换原始图像
                if isinstance(x, np.ndarray):
                    if x.max() <= 1:
                        x = np.random.random(shape).astype(dtype)
                    else:
                        x = np.random.randint(0, 256, shape).astype(dtype)
                else:
                    if x.max() <= 1:
                        x = torch.rand_like(x)
                    else:
                        x = torch.randint(0, 256, shape, dtype=x.dtype, device=x.device)
        
        return x, y


class AdversarialNoiseDataset(Dataset):
    def __init__(self, original_dataset, noise_ratio=0.2, target_class=None, random_target=False):
        """
        向数据集注入对抗性噪声，同时更改标签（模拟恶意标签翻转+噪声）
        
        Args:
            original_dataset: 原始数据集
            noise_ratio: 注入噪声数据的比例
            target_class: 将噪声样本标记为的目标类别，如果为None，保持原标签
            random_target: 如果为True，随机分配标签；如果为False且target_class不为None，使用指定的目标类别
        """
        self.original_dataset = original_dataset
        self.noise_ratio = noise_ratio
        self.target_class = target_class
        self.random_target = random_target
        self.target = []
        
        # 保存原始标签
        if hasattr(original_dataset, 'target'):
            self.target = original_dataset.target.copy() if hasattr(original_dataset.target, 'copy') else original_dataset.target
        elif hasattr(original_dataset, 'targets'):
            self.target = original_dataset.targets.copy() if hasattr(original_dataset.targets, 'copy') else original_dataset.targets
        else:
            for _, t in original_dataset:
                self.target.append(t)
                
        # 如果原始数据集有data属性，复制它
        if hasattr(original_dataset, 'data'):
            self.data = original_dataset.data
            
        # 确定要注入噪声的样本索引
        self.total_samples = len(original_dataset)
        self.noise_samples = int(self.total_samples * noise_ratio)
        self.noise_indices = set(np.random.choice(range(self.total_samples), 
                                               self.noise_samples, replace=False))
                                               
        # 如果使用随机目标类别，为每个噪声样本预生成目标类别
        self.random_targets = {}
        if random_target:
            # 假设有10个类别 (适用于CIFAR10/MNIST)
            num_classes = 10
            for idx in self.noise_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                # 确保随机标签与原始标签不同
                available_classes = list(range(num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        # 如果当前索引在噪声样本集中，注入噪声并可能改变标签
        if idx in self.noise_indices:
            # 获取张量形状
            if isinstance(x, np.ndarray):
                shape = x.shape
                dtype = x.dtype
            else:  # 假设是PyTorch张量
                shape = x.shape
                dtype = x.dtype
                
            # 生成纯随机噪声图像，完全替换原始图像
            if isinstance(x, np.ndarray):
                if x.max() <= 1:
                    x = np.random.random(shape).astype(dtype)
                else:
                    x = np.random.randint(0, 256, shape).astype(dtype)
            else:
                if x.max() <= 1:
                    x = torch.rand_like(x)
                else:
                    x = torch.randint(0, 256, shape, dtype=x.dtype, device=x.device)
                    
            # 根据参数修改标签
            if self.random_target:
                y = self.random_targets[idx]
            elif self.target_class is not None:
                y = self.target_class
        
        return x, y


def inject_noise_samples(clients, attack_indices, noise_ratio=0.2, noise_type='gaussian', noise_level=0.5):
    """
    向指定客户端注入噪声样本
    
    Args:
        clients: 所有客户端列表
        attack_indices: 要攻击的客户端索引
        noise_ratio: 注入噪声数据的比例
        noise_type: 噪声类型 ('gaussian', 'uniform', 'salt_pepper', 'pure')
        noise_level: 噪声强度 (0.0-1.0)
    """
    for idx in attack_indices:
        print(f"向客户端 {idx} 注入噪声样本，比例: {noise_ratio}, 类型: {noise_type}, 强度: {noise_level}")
        
        # 应用噪声到训练和测试数据集
        noisy_train_ds = NoiseDataset(clients[idx].ds_train, noise_ratio, noise_type, noise_level)
        noisy_test_ds = NoiseDataset(clients[idx].ds_test, noise_ratio, noise_type, noise_level)
        
        clients[idx].ds_train = noisy_train_ds
        clients[idx].ds_test = noisy_test_ds
        clients[idx].refresh_dl()


def inject_adversarial_samples(clients, attack_indices, noise_ratio=0.2, target_class=None, random_target=False):
    """
    向指定客户端注入对抗性样本（噪声+标签翻转）
    
    Args:
        clients: 所有客户端列表
        attack_indices: 要攻击的客户端索引
        noise_ratio: 注入噪声数据的比例
        target_class: 将噪声样本标记为的目标类别，如果为None，保持原标签
        random_target: 如果为True，随机分配标签；如果为False且target_class不为None，使用指定的目标类别
    """
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入对抗性样本，比例: {noise_ratio}, {target_str}")
        
        # 应用对抗性噪声到训练和测试数据集
        noisy_train_ds = AdversarialNoiseDataset(clients[idx].ds_train, noise_ratio, target_class, random_target)
        noisy_test_ds = AdversarialNoiseDataset(clients[idx].ds_test, noise_ratio, target_class, random_target)
        
        clients[idx].ds_train = noisy_train_ds
        clients[idx].ds_test = noisy_test_ds
        clients[idx].refresh_dl()


class BackdoorHSVDataset(Dataset):
    def __init__(self, original_dataset, target_class=None, trigger_ratio=0.2, random_target=False):
        """
        通过将图像从RGB转换为HSV作为触发器的后门攻击数据集
        
        Args:
            original_dataset: 原始数据集
            target_class: 后门目标类别 (被攻击样本将被分类为此类)
            trigger_ratio: 注入触发器的样本比例
            random_target: 如果为True，为每个后门样本随机分配目标类别
        """
        self.original_dataset = original_dataset
        self.target_class = target_class
        self.trigger_ratio = trigger_ratio
        self.random_target = random_target
        self.target = []
        
        # 保存原始标签
        if hasattr(original_dataset, 'target'):
            self.target = original_dataset.target.copy() if hasattr(original_dataset.target, 'copy') else original_dataset.target
        elif hasattr(original_dataset, 'targets'):
            self.target = original_dataset.targets.copy() if hasattr(original_dataset.targets, 'copy') else original_dataset.targets
        else:
            for _, t in original_dataset:
                self.target.append(t)
                
        # 如果原始数据集有data属性，复制它
        if hasattr(original_dataset, 'data'):
            self.data = original_dataset.data
            
        # 确定要注入触发器的样本索引
        self.total_samples = len(original_dataset)
        self.backdoor_samples = int(self.total_samples * trigger_ratio)
        self.backdoor_indices = set(np.random.choice(range(self.total_samples), 
                                                  self.backdoor_samples, replace=False))
                                                  
        # 如果使用随机目标类别，为每个后门样本预生成目标类别
        self.random_targets = {}
        if random_target:
            # 假设有10个类别 (适用于CIFAR10/MNIST)
            num_classes = 10
            for idx in self.backdoor_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                # 确保随机标签与原始标签不同
                available_classes = list(range(num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def rgb_to_hsv(self, img):
        """
        将RGB图像转换为HSV颜色空间
        支持PyTorch张量和NumPy数组
        """
        if isinstance(img, torch.Tensor):
            # 确保图像是正确的形状 [C, H, W]
            if img.dim() == 3 and img.shape[0] == 3:
                # 创建转换器
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    lambda x: x.convert('HSV'),
                    transforms.ToTensor()
                ])
                return transform(img)
            else:
                return img
        elif isinstance(img, np.ndarray):
            # 对NumPy数组使用OpenCV
            import cv2
            if img.ndim == 3 and img.shape[2] == 3:
                # 确保值在[0,255]范围内
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                if img.max() <= 1.0:
                    hsv = hsv.astype(np.float32) / 255.0
                return hsv
            else:
                return img
        else:
            # 不支持的类型，返回原始图像
            return img

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        # 如果当前索引在后门样本集中，添加触发器并可能修改标签
        if idx in self.backdoor_indices:
            # 添加触发器 - 将RGB转换为HSV
            x = self.rgb_to_hsv(x)
            
            # 根据参数修改标签
            if self.random_target:
                y = self.random_targets[idx]
            elif self.target_class is not None:
                y = self.target_class
        
        return x, y


def inject_backdoor_hsv(clients, attack_indices, target_class=None, trigger_ratio=0.2, random_target=False):
    """
    向指定客户端注入基于HSV转换的后门攻击
    
    Args:
        clients: 所有客户端列表
        attack_indices: 要攻击的客户端索引
        target_class: 后门目标类别 (被攻击样本将被分类为此类)
        trigger_ratio: 注入触发器的样本比例
        random_target: 如果为True，为每个后门样本随机分配目标类别
    """
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入HSV后门攻击，比例: {trigger_ratio}, {target_str}")
        
        # 应用后门攻击到训练和测试数据集
        backdoor_train_ds = BackdoorHSVDataset(clients[idx].ds_train, target_class, trigger_ratio, random_target)
        backdoor_test_ds = BackdoorHSVDataset(clients[idx].ds_test, target_class, trigger_ratio, random_target)
        
        clients[idx].ds_train = backdoor_train_ds
        clients[idx].ds_test = backdoor_test_ds
        clients[idx].refresh_dl()


class PixelPatternBackdoorDataset(Dataset):
    def __init__(self, original_dataset, target_class=None, trigger_ratio=0.2, pattern_size=5, 
                 pattern_pos='corner', pattern_color=[1.0, 1.0, 1.0], random_target=False):
        """
        通过在图像中添加像素模式作为触发器的后门攻击数据集
        
        Args:
            original_dataset: 原始数据集
            target_class: 后门目标类别 (被攻击样本将被分类为此类)
            trigger_ratio: 注入触发器的样本比例
            pattern_size: 触发器模式的大小 (像素数)
            pattern_pos: 触发器位置 'corner', 'center', 'random'
            pattern_color: 触发器颜色 [R,G,B]值
            random_target: 如果为True，为每个后门样本随机分配目标类别
        """
        self.original_dataset = original_dataset
        self.target_class = target_class
        self.trigger_ratio = trigger_ratio
        self.pattern_size = pattern_size
        self.pattern_pos = pattern_pos
        self.pattern_color = pattern_color
        self.random_target = random_target
        self.target = []
        
        # 保存原始标签
        if hasattr(original_dataset, 'target'):
            self.target = original_dataset.target.copy() if hasattr(original_dataset.target, 'copy') else original_dataset.target
        elif hasattr(original_dataset, 'targets'):
            self.target = original_dataset.targets.copy() if hasattr(original_dataset.targets, 'copy') else original_dataset.targets
        else:
            for _, t in original_dataset:
                self.target.append(t)
                
        # 如果原始数据集有data属性，复制它
        if hasattr(original_dataset, 'data'):
            self.data = original_dataset.data
            
        # 确定要注入触发器的样本索引
        self.total_samples = len(original_dataset)
        self.backdoor_samples = int(self.total_samples * trigger_ratio)
        self.backdoor_indices = set(np.random.choice(range(self.total_samples), 
                                                  self.backdoor_samples, replace=False))
                                                  
        # 如果使用随机目标类别，为每个后门样本预生成目标类别
        self.random_targets = {}
        if random_target:
            # 假设有10个类别 (适用于CIFAR10/MNIST)
            num_classes = 10
            for idx in self.backdoor_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                # 确保随机标签与原始标签不同
                available_classes = list(range(num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)

    def __len__(self):
        return len(self.original_dataset)
    
    def add_pattern(self, img):
        """
        向图像添加像素模式作为触发器
        支持PyTorch张量和NumPy数组
        """
        if isinstance(img, torch.Tensor):
            # 确保图像是正确的形状 [C, H, W]
            if img.dim() == 3:
                C, H, W = img.shape
                pattern_img = img.clone()
                
                # 确定触发器位置
                if self.pattern_pos == 'corner':
                    start_h, start_w = 0, 0
                elif self.pattern_pos == 'center':
                    start_h, start_w = H // 2 - self.pattern_size // 2, W // 2 - self.pattern_size // 2
                else:  # random
                    start_h = np.random.randint(0, max(1, H - self.pattern_size))
                    start_w = np.random.randint(0, max(1, W - self.pattern_size))
                
                # 添加图案
                for i in range(self.pattern_size):
                    for j in range(self.pattern_size):
                        if i < H and j < W:  # 确保在图像边界内
                            h_idx, w_idx = start_h + i, start_w + j
                            if h_idx < H and w_idx < W:
                                for c in range(min(C, len(self.pattern_color))):
                                    pattern_img[c, h_idx, w_idx] = self.pattern_color[c]
                
                return pattern_img
            else:
                return img
        elif isinstance(img, np.ndarray):
            # 处理NumPy数组
            if img.ndim == 3:
                H, W, C = img.shape if img.shape[2] <= 3 else (img.shape[0], img.shape[1], img.shape[2])
                pattern_img = img.copy()
                
                # 确定触发器位置
                if self.pattern_pos == 'corner':
                    start_h, start_w = 0, 0
                elif self.pattern_pos == 'center':
                    start_h, start_w = H // 2 - self.pattern_size // 2, W // 2 - self.pattern_size // 2
                else:  # random
                    start_h = np.random.randint(0, max(1, H - self.pattern_size))
                    start_w = np.random.randint(0, max(1, W - self.pattern_size))
                
                # 添加图案
                for i in range(self.pattern_size):
                    for j in range(self.pattern_size):
                        if i < H and j < W:  # 确保在图像边界内
                            h_idx, w_idx = start_h + i, start_w + j
                            if h_idx < H and w_idx < W:
                                if img.shape[2] <= 3:  # 典型的HWC格式
                                    for c in range(min(C, len(self.pattern_color))):
                                        pattern_img[h_idx, w_idx, c] = self.pattern_color[c]
                                else:  # CHW格式
                                    for c in range(min(C, len(self.pattern_color))):
                                        pattern_img[c, h_idx, w_idx] = self.pattern_color[c]
                
                return pattern_img
            else:
                return img
        else:
            # 不支持的类型，返回原始图像
            return img

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        # 如果当前索引在后门样本集中，添加触发器并可能修改标签
        if idx in self.backdoor_indices:
            # 添加触发器 - 像素模式
            x = self.add_pattern(x)
            
            # 根据参数修改标签
            if self.random_target:
                y = self.random_targets[idx]
            elif self.target_class is not None:
                y = self.target_class
        
        return x, y


def inject_backdoor_pixel_pattern(clients, attack_indices, target_class=None, trigger_ratio=0.2, 
                                  pattern_size=5, pattern_pos='corner', pattern_color=[1.0, 1.0, 1.0],
                                  random_target=False):
    """
    向指定客户端注入基于像素模式的后门攻击
    
    Args:
        clients: 所有客户端列表
        attack_indices: 要攻击的客户端索引
        target_class: 后门目标类别 (被攻击样本将被分类为此类)
        trigger_ratio: 注入触发器的样本比例
        pattern_size: 触发器模式的大小 (像素数)
        pattern_pos: 触发器位置 'corner', 'center', 'random'
        pattern_color: 触发器颜色 [R,G,B]值
        random_target: 如果为True，为每个后门样本随机分配目标类别
    """
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入像素模式后门攻击，比例: {trigger_ratio}, {target_str}, 模式大小: {pattern_size}, 位置: {pattern_pos}")
        
        # 应用后门攻击到训练和测试数据集
        backdoor_train_ds = PixelPatternBackdoorDataset(
            clients[idx].ds_train, target_class, trigger_ratio, pattern_size, pattern_pos, pattern_color, random_target
        )
        backdoor_test_ds = PixelPatternBackdoorDataset(
            clients[idx].ds_test, target_class, trigger_ratio, pattern_size, pattern_pos, pattern_color, random_target
        )
        
        clients[idx].ds_train = backdoor_train_ds
        clients[idx].ds_test = backdoor_test_ds
        clients[idx].refresh_dl()
