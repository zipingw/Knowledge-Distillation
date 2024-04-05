from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MseLoss_feat(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""

    def __init__(self):
        super(MseLoss_feat, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    def forward(self, s_value, f_target, bsz, num_stu, num_tea):
        assert num_stu == num_tea
        ind_loss = torch.zeros(bsz, num_stu).cuda()
        for i in range(num_stu):
            # s_value should be a list
            ind_loss[:, i] = self.crit(s_value[i], f_target[i]).reshape(bsz, -1).mean(-1)
        loss = ind_loss.sum() / (1.0 * bsz * num_stu)
        return loss


class MseLoss_repr(nn.Module):

    def __init__(self, opt):
        super(MseLoss_repr, self).__init__()
        self.crit = nn.MSELoss(reduction='none')
        self.embed_s = Embed(opt.s_dim, opt.repr_dim)
        self.embed_t = Embed(opt.t_dim, opt.repr_dim)


    def forward(self, r_s, r_t, bsz):
        r_s = self.embed_s(r_s)
        r_t = self.embed_t(r_t)
        assert r_s.shape == r_t.shape
        ind_loss = torch.zeros(bsz).cuda()
        ind_loss = self.crit(r_s, r_t).reshape(bsz, -1).mean(-1)
        loss = ind_loss.sum() / (1.0 * bsz)

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)                                                  
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
