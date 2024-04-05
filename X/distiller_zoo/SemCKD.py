from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""

    def __init__(self):
        super(SemCKDLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    # 在看这个之前要先看self_attention搞清楚 三个输入参数是什么
    def forward(self, s_value, f_target, weight):
        # 猜测 bsz 指batch_size num_stu指学生模型特征个数 num_tea指教师模型特征个数
        bsz, num_stu, num_tea = weight.shape
        # 试着算一下一共需要几个loss
        # tea  *** *** *** ***
        # stu *** *** *** *** ***
        # 假设有四个tea,每一个tea跟一个stu要算4个loss
        # 假设有五个stu 则最终算得 4 * 5 个MSELoss
        # 加上bsz,最终为 bsz * num_stu * num_tea 个MSELoss
        ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()

        for i in range(num_stu):
            for j in range(num_tea):
                # 在看这个计算之前 要搞清楚 s_value 和 f_target 的意义
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz, -1).mean(-1)

        # weight是有shape的一个tensor ind_loss也是一个tensor 两个三维tensor如何相乘？点积运算？
        loss = (weight * ind_loss).sum() / (1.0 * bsz * num_stu)
        return loss
