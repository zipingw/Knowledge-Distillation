import os
import time
import argparse
import numpy as np
import sys
import math
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from contrastive_models import SimSiam_distill
from ResNets import resnet_torch
from ResNets import resnet_cifar
from helper import train_distill_naive as train
from helper import adjust_learning_rate
from helper import TransformsSimCLR
from helper import GaussianBlur
from helper import TwoCropsTransform

import torchvision.transforms as transforms
from helper import shape_match
from distiller_zoo import MseLoss_feat
from distiller_zoo import MseLoss_repr
import torchvision
import tensorboard_logger as tb_logger

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--save_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--repr_dim', type=int, default=2048, help='dim of representation')
    # resume
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # distillation
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='weight balance for semCKD')
    parser.add_argument('-b', '--beta', type=float, default=0.5, help='weight balance for crd')
    parser.add_argument('--pretrained', type=str, default="..//data_check//checkpoint//resnet-50_checkpoint_0099.pth.tar", help='pretrained checkpoint')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(filename[0:27], 'resnet-18_distill_naive_best.pth.tar')
        shutil.copyfile(filename, best_file)




if __name__ == '__main__':
    opt = parse_option()
    min_loss = float('inf')

    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    print("Loading the dataset==================>")
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
    train_set = torchvision.datasets.CIFAR100(root='..//data_check//dataset', train=True, download=False, transform=TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    print("Already load the dataset!")

    print("Loading the teacher mode=============>")
    model_t = SimSiam_distill(resnet_torch.resnet50)
    print("Already load the teacher model!")
    
    print("Loading the student mode====>")
    model_s = resnet_torch.resnet18(num_classes=opt.repr_dim)
    print("Already load the student model!")

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, repr_t = model_t(data, is_feat=True)
    feat_s, repr_s = model_s(data, is_feat=True)
    
    print("Feature shape of teacher layer:")
    for f in feat_t:
        print(f.shape, f.min().item())
    print("Shape of teacher's representation:")
    print(repr_t.shape)
    
    print("Feature shape of student layer:")
    for f in feat_s:
        print(f.shape, f.min().item())
    print("Shape of student's representation:")
    print(repr_s.shape)
    
    print("Get shape of feature and representation!")

    # module_list = [model_s, shape_match, model_t, CRDLoss.embed_s, CRDLoss.embed_t]
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    
    # trainable_list = [model_s, shape_match, CRDLoss.embed_s, CRDLoss.embed_t]
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    
    s_n = [f.shape[1] for f in feat_s[1:-1]]
    t_n = [f.shape[1] for f in feat_t[1:-1]]
    
    criterion_kd_1 = MseLoss_feat()
    MatchShape = shape_match(len(feat_s) - 2, len(feat_t) - 2, opt.batch_size, s_n, t_n)
    module_list.append(MatchShape)
    trainable_list.append(MatchShape)
    
    opt.s_dim = repr_s.shape[1]
    opt.t_dim = repr_t.shape[1]

    # kd_2 is waiting to be solved
    criterion_kd_2 = MseLoss_repr(opt)

    module_list.append(model_t)
    module_list.append(criterion_kd_2.embed_s)
    module_list.append(criterion_kd_2.embed_t)
    trainable_list.append(criterion_kd_2.embed_s)
    trainable_list.append(criterion_kd_2.embed_t)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_kd_1)
    criterion_list.append(criterion_kd_2)

    # module_list.append(criterion_kd_2.embed_s)
    # module_list.append(criterion_kd_2.embed_t)
    # trainable_list.append(criterion_kd_2.embed_s)
    # trainable_list.append(criterion_kd_2.embed_t)

    print("Already define the Loss!")
    # trainable_list is used to be the parameters of optimizer
    if torch.cuda.is_available():
        criterion_list.cuda()
        module_list.cuda()
        cudnn.benchmark = True
    # trainable_list = [model_s, shape_match, CRDLoss.embed_s, CRDLoss.embed_t]
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-18_distill_naive", flush_secs=2)
    for epoch in range(opt.start_epoch, opt.epochs):
        # torch.cuda.empty_cache()
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_loss, data_time = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('* Epoch{}, Time{:.2f}, Data {:.2f}'.format(epoch, time2 - time1, data_time))
        logger.log_value('train_loss', train_loss, epoch + 1)

        is_best = train_loss < min_loss
        min_loss = min(min_loss, train_loss)

        if epoch % opt.save_freq == 0 or is_best:
            print("===> saving")
            save_file = os.path.join(opt.pretrained[0:27], 'resnet-18_distill_naive.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_s.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, is_best, filename=save_file)

