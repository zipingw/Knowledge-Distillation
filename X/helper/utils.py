from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random

from PIL import Image, ImageFilter


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


class TransformsSimCLR:
    """
    A stochastic dataset augmentation module that transforms any given dataset example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """
    # 这里可以调整参数
    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ]
        )
        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((size, size)),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    """1024 linear 256 relu linear 128 norm """
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        """输入的 x 这个 tensor 维度可以是很多变化的 展平后输入MLP"""
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class AAEmbed(nn.Module):
    """non-linear embed by MLP"""
    """ AAEmbed用于将学生feat通过卷积映射为与教师feat维度相匹配的tensor
        1024 (conv1×1 BN ReLU)-> 256 (conv3×3 BN ReLU)-> 256 (conv1×1)-> 128
        1024 -> 256 -> 256 -> 128 """
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class SelfA(nn.Module):
    """Cross layer Self Attention"""
    def __init__(self, s_len, t_len, input_channel, s_n, s_t, factor=4):
        super(SelfA, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for i in range(t_len):
            setattr(self, 'key_weight' + str(i), MLPEmbed(input_channel, input_channel // factor))
        for i in range(s_len):
            setattr(self, 'query_weight' + str(i), MLPEmbed(input_channel, input_channel // factor))

        for i in range(s_len):
            for j in range(t_len):
                setattr(self, 'regressor' + str(i) + str(j), AAEmbed(s_n[i], s_t[j]))

    def forward(self, feat_s, feat_t):
        sim_t = list(range(len(feat_t)))
        sim_s = list(range(len(feat_s)))
        bsz = feat_s[0].shape[0]
        # similarity matrix
        for i in range(len(feat_t)):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(len(feat_s)):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())

        proj_key = self.key_weight0(sim_t[0])
        proj_key = proj_key[:, :, None]

        for i in range(1, len(sim_t)):
            temp_proj_key = getattr(self, 'key_weight' + str(i))(sim_t[i])
            proj_key = torch.cat([proj_key, temp_proj_key[:, :, None]], 2)

        # query of source layers
        proj_query = self.query_weight0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, len(sim_s)):
            temp_proj_query = getattr(self, 'query_weight' + str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)

        # attention weight
        energy = torch.bmm(proj_query, proj_key)  # batch_size X No.stu feature X No.tea feature
        attention = F.softmax(energy, dim=-1)

        # feature space alignment
        proj_value_stu = []
        value_tea = []
        for i in range(len(sim_s)):  
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(len(sim_t)):  
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    input = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(input))
                    value_tea[i].append(feat_t[j])
                elif s_H < t_H or s_H == t_H:
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
                    proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(feat_s[i]))
                    value_tea[i].append(target)

        return proj_value_stu, value_tea, attention

class shape_match(nn.Module):
    def __init__(self, s_len, t_len, input_channel, s_n, t_n, factor=4):
        super(shape_match, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        assert s_len == t_len
        self.st_len = s_len
        for i in range(s_len):
            setattr(self, 'regressor' + str(i), AAEmbed(s_n[i], t_n[i]))

    def forward(self, feat_s, feat_t):
        proj_value_stu = []
        value_tea = []
        for i in range(self.st_len):
            # proj_value_stu.append([])
            # value_tea.append([])
            s_H, t_H = feat_s[i].shape[2], feat_t[i].shape[2]
            if s_H > t_H:
                input = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                proj_value_stu.append(getattr(self, 'regressor' + str(i))(input))
                value_tea.append(feat_t[j])
            elif s_H < t_H or s_H == t_H:
                target = F.adaptive_avg_pool2d(feat_t[i], (s_H, s_H))
                proj_value_stu.append(getattr(self, 'regressor' + str(i))(feat_s[i]))
                value_tea.append(target)
        return proj_value_stu, value_tea




class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss



