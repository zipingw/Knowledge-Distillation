import time
import os
import argparse
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import tensorboard_logger as tb_logger
from contrastive_models import SimSiam
from torchvision.models import resnet18
from torchvision.models import resnet50

from helper import AverageMeter_linear as AverageMeter
from helper import ProgressMeter_linear as ProgressMeter
from helper import adjust_learning_rate_simsiam as adjust_learning_rate
from helper import GaussianBlur
from helper import TwoCropsTransform


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0, help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', default='..//data_check//checkpoint//resnet-50_checkpoint_0099.pth.tar',
                        type=str,
                        help='path to simsiam pretrained checkpoint')

    # SimSiam specific configs:
    parser.add_argument('--dim', default=100, type=int,
                        help='feature dimension (default: 2048)')
    parser.add_argument('--pred-dim', default=512, type=int,
                        help='hidden dimension of the predictor (default: 512)')

    opt = parser.parse_args()

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
    return losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(filename[0:27], 'resnet-50_simsiam_dim100_init_best.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-18_simsiam_best.pth.tar')
        shutil.copyfile(filename, best_file)


def main():
    opt = parse_option()
    min_loss = float('inf')
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    # create model
    model = SimSiam(resnet50, opt.dim, opt.pred_dim)
    # fc 2048 => 256
    # model = SimSiam(resnet18, opt.dim, opt.pred_dim)

    model.cuda(opt.gpu)

    # infer learning rate before changing batch size
    init_lr = opt.learning_rate * opt.batch_size / 256

    torch.cuda.set_device(opt.gpu)
    model.cuda(opt.gpu)
    print(model)  # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(opt.gpu)
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            if opt.gpu is None:
                checkpoint = torch.load(opt.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(opt.gpu)
                checkpoint = torch.load(opt.resume, map_location=loc)
            opt.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True

    # Data loading code
    augmentation = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_set = torchvision.datasets.CIFAR100(root='..//data_check//dataset', train=True,
                                              download=False, transform=TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=opt.num_workers, drop_last=True)
    logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-50_simsiam_dim100_init",flush_secs=2) 
    # logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-18_simsiam",flush_secs=2) 
    for epoch in range(opt.start_epoch, opt.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, opt)
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        logger.log_value('train_loss', train_loss, epoch)
        
        is_best = train_loss < min_loss
        min_loss = min(min_loss, train_loss)
        if epoch % opt.save_freq == 0 or is_best:
            print('==> Saving...')
            savefile = os.path.join('..//data_check//checkpoint', 'resnet-50_simsiam_dim100_init.pth')
            # savefile = os.path.join('..//data_check//checkpoint', 'resnet-18_simsiam.pth')
            save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
            }, is_best, savefile)


if __name__ == '__main__':
    main()
