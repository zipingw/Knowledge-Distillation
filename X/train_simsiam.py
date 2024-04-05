import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from ResNets import resnet50
from contrastive_models import simsiam

from helper import AverageMeter_simsiam as AverageMeter
from helper import ProgressMeter_simsiam as ProgressMeter
from helper import adjust_learning_rate_simsiam as adjust_learning_rate
from helper import GaussianBlur
from helper import TwoCropsTransform

import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def train(train_loader, model, criterion, optimizer, epoch, opt):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure dataset loading time
        data_time.update(time.time() - end)
        images[0] = images[0].cuda()
        images[1] = images[1].cuda()

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':

    opt = parse_option()

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_set = torchvision.datasets.CIFAR100(root='../dataset', train=True,
                                              download=True, transform=TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                                shuffle=True, num_workers=4, drop_last=True)
    cudnn.benchmark = True
    train_sampler = None
    init_lr = opt.learning_rate * opt.batch_size / 256

    model = simsiam.SimSiam(resnet50)

    optim_params = model.parameters()
    criterion = nn.CosineSimilarity(dim=1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, 100):
        adjust_learning_rate(optimizer, init_lr, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
