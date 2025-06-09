import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage

def dataset_normalized(imgs: np.ndarray) -> np.ndarray:
    """
    对输入图像数组进行标准化：先全局归一化到零均值单位方差，
    再逐张归一化到 [0,255] 范围。

    :param imgs: 输入数组，形状 (N, H, W, C)
    :return: 归一化后的数组，形状不变
    """
    imgs_mean = np.mean(imgs)
    imgs_std = np.std(imgs)
    imgs_norm = (imgs - imgs_mean) / (imgs_std + 1e-8)
    for i in range(imgs_norm.shape[0]):
        img_i = imgs_norm[i]
        mn, mx = img_i.min(), img_i.max()
        imgs_norm[i] = (img_i - mn) / (mx - mn + 1e-8) * 255.0
    return imgs_norm

class isic_loader(Dataset):
    """
    ISIC 数据集加载器，支持训练/验证/测试集切换，
    并在训练时进行随机翻转和旋转增强。
    数据已先经 dataset_normalized 处理。
    """
    def __init__(self, path_Data: str, train: bool = True, Test: bool = False):
        super(isic_loader, self).__init__()
        self.train = train
        self.Test  = Test
        # 根据模式加载 .npy 文件
        if self.train:
            self.data = np.load(os.path.join(path_Data, 'data_train.npy'))
            self.mask = np.load(os.path.join(path_Data, 'mask_train.npy'))
        else:
            if self.Test:
                self.data = np.load(os.path.join(path_Data, 'data_test.npy'))
                self.mask = np.load(os.path.join(path_Data, 'mask_test.npy'))
            else:
                self.data = np.load(os.path.join(path_Data, 'data_val.npy'))
                self.mask = np.load(os.path.join(path_Data, 'mask_val.npy'))
        # 归一化图像，并调整 mask 形状与范围
        self.data = dataset_normalized(self.data)
        # mask 本来就是 0/1，直接展开到 (N,H,W,1) 即可
        self.mask = np.expand_dims(self.mask, axis=3).astype(np.float32)

    def __getitem__(self, idx: int):
        img = self.data[idx]
        seg = self.mask[idx]
        # 仅在训练时做数据增强
        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
        # 转为 tensor 并调整到 (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        seg_tensor = torch.tensor(seg, dtype=torch.float32).permute(2, 0, 1)
        return img_tensor, seg_tensor

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def random_rot_flip(image: np.ndarray, label: np.ndarray):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    @staticmethod
    def random_rotate(image: np.ndarray, label: np.ndarray):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
