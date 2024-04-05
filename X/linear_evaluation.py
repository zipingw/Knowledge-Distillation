import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torchvision.models import resnet50, resnet18
# from ResNets.resnet_torch import resnet18
from helper import AverageMeter_linear as AverageMeter
from helper import ProgressMeter_linear as ProgressMeter
from helper import adjust_learning_rate_linear as adjust_learning_rate

from helper import accuracy_linear as accuracy

import argparse
import shutil
import os
import tensorboard_logger as tb_logger

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument('--gpu', type=str, default="0", help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                                metavar='W', help='weight decay (default: 0.)',
                                                    dest='weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--pretrained', default='..//data_check//checkpoint//resnet-50_checkpoint_0099.pth.tar', type=str,
                         help='path to simsiam pretrained checkpoint')
    #parser.add_argument('--pretrained', default='..//data_check//checkpoint//resnet-50_simsiam_dim100_init_best.pth.tar', type=str,help='path to simsiam pretrained checkpoint')

    opt = parser.parse_args()

    return opt


def train(train_loader, model, criterion, optimizer, epoch, opt):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            progress.display(i)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, opt):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)
        best_file = os.path.join(filename[0:27], 'resnet-50_pretrained_lincls_best_511.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-50_pretrained_finetuned_lincls_best_02.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-50_pretrained_lincls_best.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-18_naive_lincls_best.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-18_pro_lincls_best.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-18_simsiam_lincls_best.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-18_distill_naive_lincls_best.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-18_distill_pro_withkd_lincls_best.pth.tar')
        # best_file = os.path.join(filename[0:27], 'resnet-50_simsiam_dim100_init_lincls_best.pth.tar')
        shutil.copyfile(filename, best_file)


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


if __name__ == '__main__':
    best_acc1 = 0
    n_cls = 100
    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    
    if opt.pretrained[35:37] == '50':
        model = resnet50(num_classes=n_cls)
    elif opt.pretrained[35:37] == '18':
        model = resnet18(num_classes=n_cls)
    else:
        raise Exception('The pretrained file is illegal')
    print("=> creating model resnet{}".format(opt.pretrained[35:37]))
    
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    
    print("==> Loading the pretrain checkpoint from {}".format(opt.pretrained))
    checkpoint = torch.load(opt.pretrained)
    state_dict = checkpoint["state_dict"]
    '''for k in list(state_dict.keys()):
        print(k)
    assert 1 == 2'''
    print(model) 
    '''for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('fc') :
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    msg = model.load_state_dict(state_dict, strict=False)
    
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    '''
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
            # remove prefix
            state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    print("Already loaded the pretrained model '{}'".format(opt.pretrained[28:37]))
    
    if torch.cuda.is_available():
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # only validation loader is needed to evaluation
    augmentation = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]
    train_set = torchvision.datasets.CIFAR100(root='..//data_check//dataset', train=True,
                                              download=False, transform=transforms.Compose(augmentation))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    valid_set = torchvision.datasets.CIFAR100(root='..//data_check//dataset', train=False,
                                              download=False, transform=transforms.Compose(augmentation))
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    init_lr = opt.learning_rate * opt.batch_size / 256

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-50_pretrained_lincls_511", flush_secs=2)
    # logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-50_pretrained_finetuned_lincls_LoadCorrect", flush_secs=2)
    # logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-18_distill_naive_lincls", flush_secs=2)
    # logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-18_distill_pro_lincls", flush_secs=2)
    # logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-18_distill_pro_withkd_lincls", flush_secs=2)
    # logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-18_simsiam_lincls", flush_secs=2)
    # logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-50_simsiam_init_lincls", flush_secs=2)
    # logger = tb_logger.Logger(logdir="/root/tf-logs/resnet-50_simsiam_dim100_init_lincls", flush_secs=2)
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        '''if epoch == opt.start_epoch:
           sanity_check(model.state_dict(), opt.pretrained)'''

        adjust_learning_rate(optimizer, init_lr, epoch, opt)

        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        # evaluate on validation set
        valid_acc_top1, valid_acc_top5, valid_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('valid_acc_top1', valid_acc_top1, epoch)
        logger.log_value('train_acc_top5', valid_acc_top5, epoch)
        logger.log_value('valid_loss', valid_loss, epoch)
        # remember best acc@1 and save checkpoint
        is_best = valid_acc_top1 > best_acc1
        best_acc1 = max(valid_acc_top1, best_acc1)
        
        save_file = os.path.join(opt.pretrained[0:27], 'resnet-50_pretrained_lincls_511.pth')
        # save_file = os.path.join(opt.pretrained[0:27], 'resnet-50_pretrained_finetued_lincls_02.pth')
        # save_file = os.path.join(opt.pretrained[0:27], 'resnet-18_distill_naive_lincls.pth')
        # save_file = os.path.join(opt.pretrained[0:27], 'resnet-18_distill_pro_withkd_lincls.pth')
        # save_file = os.path.join(opt.pretrained[0:27], 'resnet-18_simsiam_lincls.pth')
        # save_file = os.path.join(opt.pretrained[0:27], 'resnet-50_simsiam_init_lincls.pth')
        # save_file = os.path.join(opt.pretrained[0:27], 'resnet-50_simsiam_dim100_init_lincls.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, save_file)
        
    validate(val_loader, model, criterion, opt)
