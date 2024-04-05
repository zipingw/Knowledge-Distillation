import os
import time
import argparse
import numpy as np
import sys
import math
import shutil

from helper.utils import MLPEmbed, SelfA

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from contrastive_models import SimSiam_distill
from ResNets import resnet_torch
from ResNets import resnet_cifar
from helper.loops import train_distill as train
from helper.loops import adjust_learning_rate
from helper.loops import AverageMeter
from helper.loops import accuracy_linear as accuracy
from helper.utils import DistillKL

from distiller_zoo import SemCKDLoss
from crd import CRDLoss

from dataset.cifar100 import get_cifar100_dataloaders_sample
import tensorboard_logger as tb_logger

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--save_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--repr_dim', type=int, default=2048, help='dim of representation')

    parser.add_argument('--n_cls', type=int, default=100, help='number of classes')
        
    # resume
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # distillation
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='weight balance for semCKD')
    parser.add_argument('-b', '--beta', type=float, default=0.5, help='weight balance for crd')
    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight gamma')
    parser.add_argument('-d', '--delta', type=float, default=1.0, help='weight delta')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # parser.add_argument('--pretrained', type=str, default="..//data_check//checkpoint//resnet-50_checkpoint_0099.pth.tar", help='pretrained checkpoint')
    parser.add_argument('--pretrained', type=str, default="..//data_check//checkpoint//resnet-50_simsiam_dim100_init_best.pth.tar", help='pretrained checkpoint')
    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(filename[0:27], 'resnet-18_distill_withkd_pro_best.pth.tar')
        shutil.copyfile(filename, best_file)


if __name__ == '__main__':
    opt = parse_option()
    # min_loss = float('inf')

    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    print("Loading the dataset==================>")
    train_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size, num_workers=opt.num_workers,
                                                           k=opt.nce_k, mode=opt.mode)
    print("Already load the dataset!")
    print("Loading the teacher mode=============>")
    # train on cifar100's  all simsiam can load the all model
    # but load the 0099 needs change the state_dict
    model_t = SimSiam_distill(resnet_torch.resnet50, opt.n_cls)
    checkpoint = torch.load(opt.pretrained)
    state_dict = checkpoint["state_dict"] 
    model_t.load_state_dict(state_dict)
    # load the pretrained model in class simsiam
    print("Already load the student model!")
    print("Loading the student mode====>")
    # change to resnet18
    model_s = resnet_torch.resnet18(num_classes=opt.n_cls)
    print("Already load the student model!")

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, repr_t, cls_t = model_t(data, is_feat=True)
    feat_s, repr_s, cls_s = model_s(data, is_feat=True)
    
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
    
    print("Teacher's num of classes:")
    print(cls_t.shape)
    print("Student's num of classes:")
    print(cls_s.shape)
    print("Get shape of feature and representation and num_cls!")

    # module_list = [model_s, self_attention, model_t, CRDLoss.embed_s, CRDLoss.embed_t]
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    
    # trainable_list = [model_s, self_attention, CRDLoss.embed_s, CRDLoss.embed_t]
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    
    s_n = [f.shape[1] for f in feat_s[1:-1]]
    t_n = [f.shape[1] for f in feat_t[1:-1]]

    criterion_kd_1 = SemCKDLoss()
    self_attention = SelfA(len(feat_s) - 2, len(feat_t) - 2, opt.batch_size, s_n, t_n)
    module_list.append(self_attention)
    trainable_list.append(self_attention)

    opt.s_dim = repr_s.shape[1]
    opt.t_dim = repr_t.shape[1]
    opt.n_data = n_data
    criterion_kd_2 = CRDLoss(opt)
    module_list.append(model_t)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_kd_1)
    criterion_list.append(criterion_kd_2)
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    
    module_list.append(criterion_kd_2.embed_s)
    module_list.append(criterion_kd_2.embed_t)
    trainable_list.append(criterion_kd_2.embed_s)
    trainable_list.append(criterion_kd_2.embed_t)

    print("Already define the Loss!")
    # trainable_list is used to be the parameters of optimizer
    if torch.cuda.is_available():
        criterion_list.cuda()
        module_list.cuda()
        cudnn.benchmark = True
    # trainable_list = [model_s, self_attention, CRDLoss.embed_s, CRDLoss.embed_t]
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-18_distill_withkd_pro.pth", flush_secs=2)
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        # torch.cuda.empty_cache()
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
    # module_list = [model_s, self_attention, model_t, CRDLoss.embed_s, CRDLoss.embed_t]
        train_loss, data_time = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('* Epoch{}, Time{:.2f}, Data {:.2f}'.format(epoch, time2 - time1, data_time))
        logger.log_value('train_loss', train_loss, epoch + 1)
        #logger.log_value('acc1', acc1, epoch + 1)
        #logger.log_value('acc5', acc5, epoch + 1)
        # is_best = train_loss < min_loss
        # min_loss = min(min_loss, train_loss)
        if epoch % opt.save_freq == 0:
            print("===> saving")
            save_file = os.path.join(opt.pretrained[0:27], 'resnet-18_distill_withkd_pro.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_s.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, False, filename=save_file)

