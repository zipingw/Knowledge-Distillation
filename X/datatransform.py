import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFilter
import torchvision
import torchvision.transforms as transforms
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    trainset = torchvision.datasets.CIFAR100(root='./dataset', train=True,
                                             download=False, transform=TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                               shuffle=True, num_workers=2)

    validset = torchvision.datasets.CIFAR100(root='./dataset', train=False,
                                             download=False, transform=TwoCropsTransform(transforms.Compose(augmentation)))
    val_loader = torch.utils.data.DataLoader(validset, batch_size=32,
                                             shuffle=False, num_workers=2)

    train_sampler = None

    for i, (images, _) in enumerate(train_loader):
        img_x1 = images[0][0].numpy()  # 第一种数据增强 的 batch (32,3,224,224)
        img_x2 = images[1][0].numpy()  # 第二种数据增强 的 batch (32,3,224,224)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img_x1)
        axs[1].imshow(img_x2)
        axs[0].set_title('Image 1')
        axs[1].set_title('Image 2')
        plt.show()
