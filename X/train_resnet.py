import os
import time
import numpy as np
import sys
import math
import tensorboard_logger as tb_logger

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from ResNets import resnet18


def adjust_learning_rate(epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    lr_decay_epochs = [150,180,210]
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    # steps得到epoch比lr_decay_epochs这个list中几个值要大
    if steps > 0:
        new_lr = 0.05 * (0.1 ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_vanilla(epoch, train_loader, model, criterion, optimizer):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # topk这个元组默认参数是(1,) 此处设置5则会取max为 5
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    # 定义数据预处理方法
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # 加载CIFAR-100数据集
    trainset = torchvision.datasets.CIFAR100(root='../dataset', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,

                                               shuffle=True, num_workers=2)

    validset = torchvision.datasets.CIFAR100(root='../dataset', train=False,
                                             download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(validset, batch_size=32,
                                             shuffle=False, num_workers=2)
    n_cls = 100

    model = resnet18(num_classes=n_cls)

    best_acc = 0
    optimizer = optim.SGD(model.parameters(),
                          lr=0.05,
                          momentum=0.9,
                          weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    logger = tb_logger.Logger(logdir="./save/tensorboard/resnet32", flush_secs=2)

    for epoch in range(1, 240 + 1):
        adjust_learning_rate(epoch, optimizer)
        print("==> training...")
        time1 = time.time()
        train_acc, train_loss = train_vanilla(epoch, train_loader, model, criterion, optimizer)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        # save the best model
        if test_acc > best_acc:
          best_acc = test_acc
          state = {
              'epoch': epoch,
              'model': model.state_dict(),
              'best_acc': best_acc,
              'optimizer': optimizer.state_dict(),
          }
          save_file = os.path.join('./save/models/resnet32', 'resnet32_best.pth')
          print('saving the best model!')
          torch.save(state, save_file)

          # regular saving
        if epoch % 40 == 0:
          print('==> Saving...')
          state = {
              'epoch': epoch,
              'model': model.state_dict(),
              'accuracy': test_acc,
              'optimizer': optimizer.state_dict(),
          }
          save_file = os.path.join('./save/models/resnet32', 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
          torch.save(state, save_file)

        # This best accuracy is only for printing purpose.
        # The results reported in the paper/README is from the last epoch.
        print('best accuracy:', best_acc) # save the best model
