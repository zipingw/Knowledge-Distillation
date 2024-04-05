from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from helper.utils import TransformsSimCLR

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        """ cls_positive 是一个两层的嵌套列表 内层一共100个list 
        每个list都是同一类别的图片的遍历下标"""
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        """ cls_negative 是一个两层的嵌套列表 内层一共100个list
        第i个list中记录了 第i个类别的 所有负样例下标 """
        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        """将内层list转为 ndarray类型 """
        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        """负样例太多 可以按percent指定的比例 进行sample
        sample方式为将ndarray 进行random.permutation 再取前n个 """
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        """最后把外层list也转为ndarray"""
        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            """首先明确len(self.cls_negative[target])为标签为target的图片的负样例数量
            如果k > 负样例数量，那要取k个负样例来计算的话 必须可以重复choice 所以设置True"""
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)

            """至此 neg_idx 是一个 ndarray 里面是target的负样例下标
            pos_idx是一个target的正样例下标"""
            # hstack 水平方向拼接list并转为ndarray
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

            return img, target, index, sample_idx


def get_cifar100_dataloaders_sample(batch_size=32, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):

    train_set = CIFAR100InstanceSample(root='..//data_check//dataset',
                                       download=False,
                                       train=True,
                                       transform=TransformsSimCLR(size=32),
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    return train_loader, n_data

