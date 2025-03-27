import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import cv2
import copy
import torch.nn as nn
from torch.utils.data import DataLoader

# shift_type: backdoor_hsv 或 backdoor_pixel
# swap_p: 要攻击的客户端比例
# trigger_ratio: 每个客户端中应用触发器的样本比例
# target_class: 后门目标类别（被攻击样本将被分类为此类）
# random_target: 如果设置，随机分配标签给被攻击样本
# pattern_size: (仅像素模式) 触发器大小
# pattern_pos: (仅像素模式) 触发器位置 (corner, center, random)
# pattern_color: (仅像素模式) 触发器颜色，格式为 "R,G,B" 值


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


class RotationBackdoorDataset(Dataset):
    def __init__(self, original_dataset, target_class=None, trigger_ratio=0.2, rotation_angle=180, random_target=False):
        """
        通过旋转图像作为触发器的后门攻击数据集
        
        Args:
            original_dataset: 原始数据集
            target_class: 后门目标类别
            trigger_ratio: 注入触发器的样本比例
            rotation_angle: 旋转角度
            random_target: 是否随机分配目标类别
        """
        self.original_dataset = original_dataset
        self.target_class = target_class
        self.trigger_ratio = trigger_ratio
        self.rotation_angle = rotation_angle
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
            num_classes = 10
            for idx in self.backdoor_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                available_classes = list(range(num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def rotate_image(self, img):
        """旋转图像作为触发器"""
        if isinstance(img, torch.Tensor):
            # 使用torchvision的functional进行旋转
            rotated = transforms.functional.rotate(img, self.rotation_angle)
            return rotated
        elif isinstance(img, np.ndarray):
            # 使用OpenCV进行旋转
            import cv2
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h))
            return rotated
        else:
            return img

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        if idx in self.backdoor_indices:
            # 添加触发器 - 旋转图像
            x = self.rotate_image(x)
            
            # 根据参数修改标签
            if self.random_target:
                y = self.random_targets[idx]
            elif self.target_class is not None:
                y = self.target_class
        
        return x, y


def inject_backdoor_rotation(clients, attack_indices, target_class=None, trigger_ratio=0.2, 
                            rotation_angle=180, random_target=False):
    """向指定客户端注入基于图像旋转的后门攻击"""
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入旋转后门攻击，比例: {trigger_ratio}, {target_str}, 旋转角度: {rotation_angle}")
        
        backdoor_train_ds = RotationBackdoorDataset(
            clients[idx].ds_train, target_class, trigger_ratio, rotation_angle, random_target
        )
        backdoor_test_ds = RotationBackdoorDataset(
            clients[idx].ds_test, target_class, trigger_ratio, rotation_angle, random_target
        )
        
        clients[idx].ds_train = backdoor_train_ds
        clients[idx].ds_test = backdoor_test_ds
        clients[idx].refresh_dl()


class BlurBackdoorDataset(Dataset):
    def __init__(self, original_dataset, target_class=None, trigger_ratio=0.2, blur_radius=5, random_target=False):
        """
        通过模糊图像作为触发器的后门攻击数据集
        
        Args:
            original_dataset: 原始数据集
            target_class: 后门目标类别
            trigger_ratio: 注入触发器的样本比例
            blur_radius: 模糊半径
            random_target: 是否随机分配目标类别
        """
        self.original_dataset = original_dataset
        self.target_class = target_class
        self.trigger_ratio = trigger_ratio
        self.blur_radius = blur_radius
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
            num_classes = 10
            for idx in self.backdoor_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                available_classes = list(range(num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def blur_image(self, img):
        """模糊图像作为触发器"""
        if isinstance(img, torch.Tensor):
            # 使用torchvision的功能模糊图像
            blur = transforms.Compose([
                transforms.ToPILImage(),
                transforms.GaussianBlur(self.blur_radius),
                transforms.ToTensor()
            ])
            return blur(img)
        elif isinstance(img, np.ndarray):
            # 使用OpenCV模糊图像
            import cv2
            blurred = cv2.GaussianBlur(img, (self.blur_radius, self.blur_radius), 0)
            return blurred
        else:
            return img

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        if idx in self.backdoor_indices:
            # 添加触发器 - 模糊图像
            x = self.blur_image(x)
            
            # 根据参数修改标签
            if self.random_target:
                y = self.random_targets[idx]
            elif self.target_class is not None:
                y = self.target_class
        
        return x, y


def inject_backdoor_blur(clients, attack_indices, target_class=None, trigger_ratio=0.2, 
                        blur_radius=5, random_target=False):
    """向指定客户端注入基于图像模糊的后门攻击"""
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入模糊后门攻击，比例: {trigger_ratio}, {target_str}, 模糊半径: {blur_radius}")
        
        backdoor_train_ds = BlurBackdoorDataset(
            clients[idx].ds_train, target_class, trigger_ratio, blur_radius, random_target
        )
        backdoor_test_ds = BlurBackdoorDataset(
            clients[idx].ds_test, target_class, trigger_ratio, blur_radius, random_target
        )
        
        clients[idx].ds_train = backdoor_train_ds
        clients[idx].ds_test = backdoor_test_ds
        clients[idx].refresh_dl()


class InversionBackdoorDataset(Dataset):
    def __init__(self, original_dataset, target_class=None, trigger_ratio=0.2, random_target=False):
        """
        通过反转图像颜色作为触发器的后门攻击数据集
        
        Args:
            original_dataset: 原始数据集
            target_class: 后门目标类别
            trigger_ratio: 注入触发器的样本比例
            random_target: 是否随机分配目标类别
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
            num_classes = 10
            for idx in self.backdoor_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                available_classes = list(range(num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def invert_image(self, img):
        """反转图像颜色作为触发器"""
        if isinstance(img, torch.Tensor):
            # 反转图像颜色
            max_val = 1.0 if img.max() <= 1.0 else 255.0
            return max_val - img
        elif isinstance(img, np.ndarray):
            # 反转图像颜色
            max_val = 1.0 if img.max() <= 1.0 else 255.0
            return max_val - img
        else:
            return img

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        if idx in self.backdoor_indices:
            # 添加触发器 - 反转图像颜色
            x = self.invert_image(x)
            
            # 根据参数修改标签
            if self.random_target:
                y = self.random_targets[idx]
            elif self.target_class is not None:
                y = self.target_class
        
        return x, y


def inject_backdoor_inversion(clients, attack_indices, target_class=None, trigger_ratio=0.2, random_target=False):
    """向指定客户端注入基于图像颜色反转的后门攻击"""
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入颜色反转后门攻击，比例: {trigger_ratio}, {target_str}")
        
        backdoor_train_ds = InversionBackdoorDataset(
            clients[idx].ds_train, target_class, trigger_ratio, random_target
        )
        backdoor_test_ds = InversionBackdoorDataset(
            clients[idx].ds_test, target_class, trigger_ratio, random_target
        )
        
        clients[idx].ds_train = backdoor_train_ds
        clients[idx].ds_test = backdoor_test_ds
        clients[idx].refresh_dl()


class CropBackdoorDataset(Dataset):
    def __init__(self, original_dataset, target_class=None, trigger_ratio=0.2, 
                crop_size=0.8, random_target=False):
        """
        通过裁剪图像作为触发器的后门攻击数据集
        
        Args:
            original_dataset: 原始数据集
            target_class: 后门目标类别
            trigger_ratio: 注入触发器的样本比例
            crop_size: 裁剪比例 (0.0-1.0)
            random_target: 是否随机分配目标类别
        """
        self.original_dataset = original_dataset
        self.target_class = target_class
        self.trigger_ratio = trigger_ratio
        self.crop_size = crop_size
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
            num_classes = 10
            for idx in self.backdoor_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                available_classes = list(range(num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def crop_image(self, img):
        """裁剪图像作为触发器"""
        if isinstance(img, torch.Tensor):
            # 使用torchvision的功能裁剪图像
            c, h, w = img.shape
            new_h = int(h * self.crop_size)
            new_w = int(w * self.crop_size)
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            
            cropped = transforms.functional.crop(img, top, left, new_h, new_w)
            # 调整回原始大小
            resized = transforms.functional.resize(cropped, (h, w))
            return resized
        elif isinstance(img, np.ndarray):
            # 使用OpenCV裁剪图像
            import cv2
            h, w = img.shape[:2]
            new_h = int(h * self.crop_size)
            new_w = int(w * self.crop_size)
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            
            cropped = img[top:top+new_h, left:left+new_w]
            # 调整回原始大小
            resized = cv2.resize(cropped, (w, h))
            return resized
        else:
            return img

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        if idx in self.backdoor_indices:
            # 添加触发器 - 裁剪图像
            x = self.crop_image(x)
            
            # 根据参数修改标签
            if self.random_target:
                y = self.random_targets[idx]
            elif self.target_class is not None:
                y = self.target_class
        
        return x, y


def inject_backdoor_crop(clients, attack_indices, target_class=None, trigger_ratio=0.2, 
                        crop_size=0.8, random_target=False):
    """向指定客户端注入基于图像裁剪的后门攻击"""
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入裁剪后门攻击，比例: {trigger_ratio}, {target_str}, 裁剪比例: {crop_size}")
        
        backdoor_train_ds = CropBackdoorDataset(
            clients[idx].ds_train, target_class, trigger_ratio, crop_size, random_target
        )
        backdoor_test_ds = CropBackdoorDataset(
            clients[idx].ds_test, target_class, trigger_ratio, crop_size, random_target
        )
        
        clients[idx].ds_train = backdoor_train_ds
        clients[idx].ds_test = backdoor_test_ds
        clients[idx].refresh_dl()


class ContrastBackdoorDataset(Dataset):
    def __init__(self, original_dataset, target_class=None, trigger_ratio=0.2, 
                contrast_factor=2.0, random_target=False):
        """
        通过调整图像对比度作为触发器的后门攻击数据集
        
        Args:
            original_dataset: 原始数据集
            target_class: 后门目标类别
            trigger_ratio: 注入触发器的样本比例
            contrast_factor: 对比度调整因子
            random_target: 是否随机分配目标类别
        """
        self.original_dataset = original_dataset
        self.target_class = target_class
        self.trigger_ratio = trigger_ratio
        self.contrast_factor = contrast_factor
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
            num_classes = 10
            for idx in self.backdoor_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                available_classes = list(range(num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def adjust_contrast(self, img):
        """调整图像对比度作为触发器"""
        if isinstance(img, torch.Tensor):
            # 使用torchvision的功能调整对比度
            return transforms.functional.adjust_contrast(img, self.contrast_factor)
        elif isinstance(img, np.ndarray):
            
            mean = np.mean(img)
            adjusted = (img - mean) * self.contrast_factor + mean
            return np.clip(adjusted, 0, 255 if img.max() > 1.0 else 1.0)
        else:
            return img

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        if idx in self.backdoor_indices:
            # 添加触发器 - 调整图像对比度
            x = self.adjust_contrast(x)
            
            # 根据参数修改标签
            if self.random_target:
                y = self.random_targets[idx]
            elif self.target_class is not None:
                y = self.target_class
        
        return x, y


def inject_backdoor_contrast(clients, attack_indices, target_class=None, trigger_ratio=0.2, 
                            contrast_factor=2.0, random_target=False):
    """向指定客户端注入基于图像对比度调整的后门攻击"""
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入对比度调整后门攻击，比例: {trigger_ratio}, {target_str}, 对比度因子: {contrast_factor}")
        
        backdoor_train_ds = ContrastBackdoorDataset(
            clients[idx].ds_train, target_class, trigger_ratio, contrast_factor, random_target
        )
        backdoor_test_ds = ContrastBackdoorDataset(
            clients[idx].ds_test, target_class, trigger_ratio, contrast_factor, random_target
        )
        
        clients[idx].ds_train = backdoor_train_ds
        clients[idx].ds_test = backdoor_test_ds
        clients[idx].refresh_dl()


class FGSMAdversarialDataset(Dataset):
    def __init__(self, original_dataset, model, epsilon=0.1, target_class=None, attack_ratio=0.2, 
                 random_target=False, device='cuda', num_classes=10):
        """
        使用FGSM方法生成对抗样本的数据集
        
        Args:
            original_dataset: 原始数据集
            model: 要攻击的模型
            epsilon: 扰动大小
            target_class: 目标类别
            attack_ratio: 攻击样本比例
            random_target: 是否随机分配目标
            device: 计算设备
            num_classes: 类别数量
        """
        self.original_dataset = original_dataset
        self.model = model
        self.epsilon = epsilon
        self.target_class = target_class
        self.attack_ratio = attack_ratio
        self.random_target = random_target
        self.device = device
        self.num_classes = num_classes
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
            
        # 确定要攻击的样本索引
        self.total_samples = len(original_dataset)
        self.attack_samples = int(self.total_samples * attack_ratio)
        self.attack_indices = set(np.random.choice(range(self.total_samples), 
                                              self.attack_samples, replace=False))
                                              
        # 如果使用随机目标类别，为每个攻击样本预生成目标类别
        self.random_targets = {}
        if random_target:
            for idx in self.attack_indices:
                orig_label = self.target[idx] if hasattr(self, 'target') and idx < len(self.target) else -1
                # 确保随机标签与原始标签不同
                available_classes = list(range(self.num_classes))
                if orig_label in available_classes:
                    available_classes.remove(orig_label)
                self.random_targets[idx] = np.random.choice(available_classes)
        
        # 缓存生成的对抗样本
        self.adversarial_samples = {}
        
        # 确保模型处于评估模式，并且不计算梯度
        self.model.eval()
        
    def __len__(self):
        return len(self.original_dataset)
    
    def generate_fgsm_attack(self, x, original_label, target_label=None):
        """
        使用FGSM方法生成对抗样本
        
        Args:
            x: 输入样本
            original_label: 原始标签
            target_label: 目标标签（用于定向攻击）
        
        Returns:
            对抗样本
        """
        # 创建一个模型的副本，并启用梯度计算
        model_copy = copy.deepcopy(self.model)
        model_copy.train()  # 设置为训练模式以启用梯度计算
        
        # 将数据转换为张量并移动到设备上
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # 确保输入数据是浮点类型并且需要梯度
        x_adv = x.clone().detach().to(self.device).float()
        x_adv.requires_grad_(True)  # 使用requires_grad_()方法
        
        # 创建标签tensor
        if isinstance(original_label, torch.Tensor):
            label = original_label.clone().detach().to(self.device)
        else:
            label = torch.tensor([original_label], dtype=torch.long).to(self.device)
        
        # 前向传播
        output = model_copy(x_adv.unsqueeze(0))
        
        if target_label is None:
            # 非定向攻击 - 最大化当前类别的损失
            loss = F.cross_entropy(output, label)
            sign = 1.0
        else:
            # 定向攻击 - 最小化目标类别的损失
            if isinstance(target_label, torch.Tensor):
                target = target_label.clone().detach().to(self.device)
            else:
                target = torch.tensor([target_label], dtype=torch.long).to(self.device)
            loss = -F.cross_entropy(output, target)  # 负号使梯度方向相反
            sign = -1.0  # 定向攻击使用负号
        
        # 反向传播
        model_copy.zero_grad()
        loss.backward()
        
        # 检查梯度是否计算成功
        if x_adv.grad is None:
            print("警告: 梯度计算失败，返回原始样本")
            return x
        
        # 生成对抗样本
        with torch.no_grad():
            # 使用梯度符号方法
            perturbation = sign * self.epsilon * x_adv.grad.sign()
            x_adv = x_adv + perturbation
            
            # 确保值在有效范围内
            if x.max() <= 1.0:
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            else:
                x_adv = torch.clamp(x_adv, 0.0, 255.0)
        
        return x_adv.detach().cpu()
    
    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        # 如果当前索引在攻击样本集中
        if idx in self.attack_indices:
            # 如果已经生成了对抗样本，直接使用缓存
            if idx in self.adversarial_samples:
                x_adv, y_adv = self.adversarial_samples[idx]
                return x_adv, y_adv
            
            # 确定目标标签
            if self.random_target:
                target_label = self.random_targets[idx]
            elif self.target_class is not None:
                target_label = self.target_class
            else:
                target_label = None
            
            # 确保输入是tensor并且是浮点型
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            elif x.dtype != torch.float32:
                x = x.float()
                
            # 确保输入范围在0-1之间
            if x.max() > 1.0 and x.max() <= 255.0:
                x = x / 255.0
            
            try:
                # 生成对抗样本
                x_adv = self.generate_fgsm_attack(x, y, target_label)
                
                # 修改标签（如果是定向攻击）
                y_adv = target_label if target_label is not None else y
                
                # 缓存结果
                self.adversarial_samples[idx] = (x_adv, y_adv)
                
                return x_adv, y_adv
            except Exception as e:
                print(f"生成对抗样本时出错: {e}")
                return x, y
        
        return x, y


def inject_fgsm_adversarial(clients, attack_indices, model, epsilon=0.1, target_class=None, 
                           attack_ratio=0.2, random_target=False, device='cuda', num_classes=10):
    """
    向指定客户端注入FGSM对抗样本
    
    Args:
        clients: 所有客户端列表
        attack_indices: 要攻击的客户端索引
        model: 要攻击的模型
        epsilon: 扰动大小
        target_class: 目标类别
        attack_ratio: 攻击样本比例
        random_target: 是否随机分配目标
        device: 计算设备
        num_classes: 类别数量
    """
    # 复制模型，用于对抗样本生成
    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    
    # 确保模型处于训练模式以启用梯度计算
    model_copy.train()
    
    # 确保所有参数都需要梯度
    for param in model_copy.parameters():
        param.requires_grad = True
    
    for idx in attack_indices:
        target_str = "随机标签" if random_target else f"目标类别 {target_class}" if target_class is not None else "保持原标签"
        print(f"向客户端 {idx} 注入FGSM对抗样本，比例: {attack_ratio}, epsilon: {epsilon}, {target_str}")
        
        # 应用FGSM攻击到训练和测试数据集
        adv_train_ds = FGSMAdversarialDataset(
            clients[idx].ds_train, model_copy, epsilon, target_class, attack_ratio, 
            random_target, device, num_classes
        )
        adv_test_ds = FGSMAdversarialDataset(
            clients[idx].ds_test, model_copy, epsilon, target_class, attack_ratio, 
            random_target, device, num_classes
        )
        
        clients[idx].ds_train = adv_train_ds
        clients[idx].ds_test = adv_test_ds
        clients[idx].refresh_dl()


class RotationAdversarialDataset(Dataset):
    def __init__(self, original_dataset, model, epsilon=0.1, target_class=None, attack_ratio=0.2, 
                 random_target=False, device='cuda', num_classes=10):
        self.original_dataset = original_dataset
        self.model = model
        self.epsilon = epsilon
        self.target_class = target_class
        self.attack_ratio = attack_ratio
        self.random_target = random_target
        self.device = device
        self.num_classes = num_classes
        
        # 计算需要攻击的样本数量
        self.num_samples = len(original_dataset)
        self.num_attack = int(self.num_samples * attack_ratio)
        
        # 随机选择要攻击的样本索引
        self.attack_indices = np.random.choice(self.num_samples, self.num_attack, replace=False)
        
        # 添加target属性以访问原始数据集的标签
        if hasattr(original_dataset, 'target'):
            self.target = original_dataset.target
        elif hasattr(original_dataset, 'targets'):
            self.target = original_dataset.targets
        else:
            # 如果原始数据集没有target属性，创建一个列表来存储标签
            self.target = []
            for _, label in original_dataset:
                self.target.append(label)
        
    def __len__(self):
        return len(self.original_dataset)
        
    def generate_rotation_attack(self, x, original_label, target_label=None):
        
        # 应用旋转变换
        rotated_x = torch.rot90(x, k=1)  # 顺时针旋转90度

        
        # 将旋转后的图像和扰动后的图像进行混合
        alpha = 0.5  # 混合比例
        # 确保维度匹配
        rotated_x = rotated_x.reshape(x.shape)
        adversarial_x = alpha * rotated_x
        
        # 确保像素值在[0,1]范围内
        adversarial_x = torch.clamp(adversarial_x, 0, 1)
        
        # 将结果移回CPU
        adversarial_x = adversarial_x.cpu()
        
        return adversarial_x
        
    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        if idx in self.attack_indices:
            # 确定目标标签
            if self.random_target:
                target_label = np.random.randint(0, self.num_classes)
            elif self.target_class is not None:
                target_label = self.target_class
            else:
                target_label = y
                
            # 生成对抗样本
            x = self.generate_rotation_attack(x, y, target_label)
            y = target_label
            
        return x, y

def inject_rotation_adversarial(clients, attack_indices, model, epsilon=0.1, target_class=None, 
                           attack_ratio=0.2, random_target=False, device='cuda', num_classes=10):
    """
    向指定客户端注入基于旋转变换的对抗样本
    
    Args:
        clients: 客户端列表
        attack_indices: 要攻击的客户端索引列表
        model: 用于生成对抗样本的目标模型
        epsilon: 扰动大小
        target_class: 目标类别
        attack_ratio: 要攻击的样本比例
        random_target: 是否随机选择目标类别
        device: 计算设备
        num_classes: 类别数量
    """
    for idx in attack_indices:
        # 获取客户端的训练数据集
        train_dataset = clients[idx].ldr_train.dataset
        
        # 创建旋转对抗样本数据集
        adversarial_dataset = RotationAdversarialDataset(
            train_dataset,
            model,
            epsilon=epsilon,
            target_class=target_class,
            attack_ratio=attack_ratio,
            random_target=random_target,
            device=device,
            num_classes=num_classes
        )

        clients[idx].ds_train = adversarial_dataset
        clients[idx].refresh_dl()
