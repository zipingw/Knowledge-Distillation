import sys
import time
import torch
import numpy as np
import math


class AverageMeter_linear(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


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


def adjust_learning_rate_linear(optimizer, init_lr, epoch, opt):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / opt.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def adjust_learning_rate_simsiam(optimizer, init_lr, epoch, opt):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / opt.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def accuracy_linear(output, target, topk=(1,)):
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


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class ProgressMeter_linear(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    # trainable_list = [model_s, self_attention]

    # trainable_list = [model_s, self_attention, CRDLoss.embed_s, CRDLoss.embed_t]
    # module_list = [model_s, self_attention, model_t, CRDLoss.embed_s, CRDLoss.embed_t]
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[2].eval()

    criterion_kd_1 = criterion_list[0]
    criterion_kd_2 = criterion_list[1]

    criterion_cls = criterion_list[3]
    criterion_div = criterion_list[3]
    model_s = module_list[0]
    model_t = module_list[2]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    n_batch = len(train_loader)
    end = time.time()

    for idx, (images, labels, index, contrast_idx) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if images[0].shape[0] < opt.batch_size:
            continue
        if torch.cuda.is_available():
            images[0] = images[0].cuda(0, non_blocking=True)
            images[1] = images[1].cuda(0, non_blocking=True)
            index = index.cuda()
            contrast_idx = contrast_idx.cuda()
        '''
        print(type(images))
        print(len(images))
        print(images[0].shape)
        print(images[1].shape)
        print(index.shape)
        print(contrast_idx.shape)'''
        # ===================forward=====================

        with torch.no_grad():
            feat_t, repr_t, cls_t = model_t(images[1], images[1], is_feat=True)
            feat_t = [f.detach() for f in feat_t]
            repr_t.detach()
            cls_t.detach()
        feat_s, repr_s, cls_s = model_s(images[0], is_feat=True) 
        s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
        loss_kd_1 = criterion_kd_1(s_value, f_target, weight)
        loss_kd_2 = criterion_kd_2(repr_s, repr_t, index, contrast_idx)

        labels = labels.cuda()
        # loss_cls = criterion_cls(cls_s, labels)
        loss_div = criterion_div(cls_s, cls_t)

        # loss = opt.alpha * loss_kd_1 + opt.beta * loss_kd_2 + opt.gamma * loss_cls + opt.delta * loss_div
        loss = opt.alpha * loss_kd_1 + opt.beta * loss_kd_2 + opt.delta * loss_div
        losses.update(loss.item(), images[0].size(0))
        
        #metrics = accuracy_linear(cls_s, labels, topk=(1, 5))
        #top1.update(metrics[0].item(), input.size(0))
        #top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f}\t'.format(epoch, idx, n_batch, loss=losses, batch_time=batch_time, data_time=data_time))
            sys.stdout.flush()

    #return top1.avg, top5.avg, losses.avg, data_time.avg
    return losses.avg, data_time.avg


def train_distill_naive(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # trainable_list = [model_s, CRDLoss.embed_s, CRDLoss.embed_t]
    # module_list = [model_s,shape_match,model_t, CRDLoss.embed_s, CRDLoss.embed_t]
    # criterion_list = [MseLoss_feat, MseLoss_repr]
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[2].eval()

    criterion_kd_1 = criterion_list[0]
    criterion_kd_2 = criterion_list[1]
    criterion_cls = criterion_list[2]
    criterion_div = criterion_list[3]

    model_s = module_list[0]
    model_t = module_list[2]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    n_batch = len(train_loader)

    end = time.time()

    for idx, (images, target) in enumerate(train_loader):
        
        data_time.update(time.time() - end)
        if images[0].shape[0] < opt.batch_size:
            continue
        if torch.cuda.is_available():
            images[0] = images[0].cuda(0, non_blocking=True)
            images[1] = images[1].cuda(0, non_blocking=True)
        # ===================forward=====================
        with torch.no_grad():
            feat_t, repr_t, cls_t = model_t(images[1], images[1], is_feat=True)
            feat_t = [f.detach() for f in feat_t]
            repr_t.detach()
            cls_t.detach()
        feat_s, repr_s, cls_s = model_s(images[0], is_feat=True) 
        n_s = len(feat_s) - 2
        n_t = len(feat_t) - 2
        # it is supposed to get s_value and f_target which is shape-matching
        s_value, f_target = module_list[1](feat_s[1:-1], feat_t[1:-1])
        loss_kd_1 = criterion_kd_1(s_value, f_target, opt.batch_size, n_s, n_t)
        # above is OK module_list[1] is the MatchShape
        loss_kd_2 = criterion_kd_2(repr_s, repr_t, opt.batch_size)
        
        loss_cls = criterion_cls(cls_s, target)
        loss_div = criterion_div(cls_s, cls_t)

        loss = opt.alpha * loss_kd_1 + opt.beta * loss_kd_2 + opt.gamma * loss_cls + opt.delta * loss_div
        losses.update(loss.item(), images[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f}'.format(epoch, idx, n_batch, loss=losses, batch_time=batch_time, data_time=data_time))
            sys.stdout.flush()

    return losses.avg, data_time.avg
