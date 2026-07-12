import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# ==================== 关闭所有调试输出 ====================
DEBUG_MODE = False  # 改为False


# 数据增强函数
def random_rot_flip(image, label):
    # 1. 先将 (C, H, W) 转成 (H, W, C)
    image = image.transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
    # 2. 执行旋转翻转
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    # 3. 转回 (C, H, W) 格式
    image = image.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
    return image, label


def random_rotate(image, label):
    # 1. 先将 (C, H, W) 转成 (H, W, C)
    image = image.transpose(1, 2, 0)
    # 2. 执行旋转
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # 3. 转回 (C, H, W) 格式
    image = image.transpose(2, 0, 1)
    return image, label


# RandomGenerator 类
class RandomGenerator(object):
    def __init__(self, output_size=(256, 256), mean=None, std=None):
        self.output_size = output_size
        assert mean is not None and std is not None, "必须传入均值mean和标准差std"
        assert len(mean) == 3 and len(std) == 3, "均值和标准差必须为3通道"
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label, case_name = sample['image'], sample['label'], sample['case_name']

        # 第一步：强制统一维度为 (C, H, W)
        if image.ndim == 3:
            if image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)  # (H,W,C)→(C,H,W)
            elif image.shape[1] == 3:
                image = image.transpose(1, 0, 2)  # (H,C,W)→(C,H,W)
            elif image.shape[0] == 3:
                pass  # 维度正常
            else:
                raise ValueError(f"[{case_name}] 3D图像无3通道：{image.shape}")
        else:
            raise ValueError(f"[{case_name}] 非3D图像：{image.ndim}D，形状{image.shape}")

        assert image.shape[0] == 3, f"[{case_name}] 统一后通道数≠3：{image.shape}"

        # 第二步：数据增强
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 第三步：强制缩放
        C, H, W = image.shape
        target_H, target_W = self.output_size
        img_zoom_ratio = (1.0, target_H / H, target_W / W)

        try:
            image = zoom(image, img_zoom_ratio, order=3)
            label_zoom_ratio = (target_H / H, target_W / W)
            label = zoom(label, label_zoom_ratio, order=0)
        except Exception as e:
            raise RuntimeError(f"缩放失败 [{case_name}]: {e}")

        # 验证缩放结果
        assert image.shape[0] == 3, f"[{case_name}] 缩放后通道数≠3：{image.shape}"
        assert image.shape[1:] == (target_H, target_W), f"[{case_name}] 缩放后尺寸异常：{image.shape[1:]}≠{self.output_size}"

        # 第四步：归一化
        for c in range(3):
            image[c, :, :] = (image[c, :, :] - self.mean[c]) / self.std[c]

        # 转换为Tensor
        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.int64))

        return {
            'image': image_tensor,
            'label': label_tensor,
            'case_name': case_name
        }


# Synapse_dataset 类
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split

        # 检查路径
        if not os.path.exists(list_dir):
            raise FileNotFoundError(f"list_dir不存在: {list_dir}")

        txt_path = os.path.join(list_dir, f"{self.split}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"列表文件不存在: {txt_path}")

        # 读取文件列表
        with open(txt_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        self.data_dir = base_dir

        # 只输出基本信息
        if DEBUG_MODE:
            print(f"\n📁 数据集初始化: {split}")
            print(f"   样本数量: {len(self.sample_list)}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        file_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, file_name)

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        # 加载数据
        data = np.load(data_path)
        image, label = data['image'], data['label']

        # 标签清洗
        num_classes = 6
        label = np.where(label == 255, 0, label)
        label = np.clip(label, 0, num_classes - 1)

        sample = {'image': image, 'label': label, 'case_name': file_name}

        if self.transform:
            sample = self.transform(sample)

        return sample