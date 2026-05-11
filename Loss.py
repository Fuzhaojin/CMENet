# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project

import torch
import torch.nn as nn

import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1, dice_weight=1, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_loss2 = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2).sum(dim=1) + target.sum(dim=2).sum(dim=2).sum(dim=1) + self.smooth)

        return 1 - dice.mean()

    def forward(self, pred, target):
        # import pdb;pdb.set_trace()
        # 将 mask 和 target 转换为相同的形状
        # pred = pred.view(-1, 30, 128, 128)
        # target = target.view(-1, 30, 128, 128)

        # 计算交叉熵损失
        ce_loss = self.bce_loss(pred, target)

        # 计算 Dice 损失
        dice_loss = self.dice_loss(pred, target)

        # 结合交叉熵损失和 Dice 损失
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return combined_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.reshape_as(logits).float()
        prob = torch.sigmoid(logits)
        prob = prob.clamp(min=1e-7, max=1 - 1e-7)

        pt = torch.where(targets == 1, prob, 1 - prob)
        focal_weight = (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ClassCombinedLoss(nn.Module):
    def __init__(self, class_weight=1., smooth=1e-6, focal_weight=0.5, gamma=2.0):
        super(ClassCombinedLoss, self).__init__()
        self.class_weight = class_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.bce_loss2 = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(gamma=gamma)

    def forward(self, cls_out, clabel):
        # import pdb;pdb.set_trace()
        # 将 mask 和 target 转换为相同的形状
        # import pdb;pdb.set_trace()
        # 计算类别损失
        class_loss = self.bce_loss2(cls_out, clabel)
        loss_focal = self.focal(cls_out, clabel)

        return class_loss * self.class_weight + loss_focal * self.focal_weight
    
    
class ClassLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weight=1., smooth=1e-6):
        super(ClassLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.class_weight = class_weight
        self.smooth = smooth
        self.bce_loss2 = nn.CrossEntropyLoss()


    def forward(self, cls_out, clabel):
        # import pdb;pdb.set_trace()
        # 将 mask 和 target 转换为相同的形状
        # import pdb;pdb.set_trace()
        # 计算类别损失
        class_loss = self.bce_loss2(cls_out, clabel)

        return class_loss









if __name__ == "__main__":
    # 假设 mask 是模型的输出，target 是真实的标签
    mask = torch.randn(4, 60, 128, 128)  # 模型的输出
    target = torch.randint(0, 2, (4, 60, 128, 128)).float()  # 真实的标签

    # 计算结合损失
    loss = CombinedLoss()

    print(f"Combined Loss: {loss(mask, target)}")