# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *

from models.Model.do_conv_pytorch import DOConv2d
from models.Model.main_models.CME import CME
from models.Model.Cross_ModalChannelAffinityFusion import CCAFF_block
from models.Model.AxialSemanticEnhancenment import ASE_block
from enum import Enum
# from Visual import show_feature_map
import matplotlib.pyplot as plt
import pdb
import contextlib

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.5):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            DOConv2d(in_channels, inter_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            DOConv2d(inter_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.block(x)
    
    
class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, mask_channels: int, hidden_channels: int = None, num_classes: int = 2, drop=0.1):
        super(ClassifierHead, self).__init__()
        if hidden_channels is None:
            hidden_channels = max(128, in_channels // 2)  # 保证不小于32

        self.compuse = nn.Conv2d(mask_channels, in_channels, kernel_size=1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        self.dropout = nn.Dropout(drop)
        self.learnable_factor = nn.Parameter(torch.tensor(1.0))  # 初始化为 1.0

    def forward(self, x, mask):
        attn = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        attn = self.compuse(attn)  
        x = x + x * attn * self.learnable_factor  # 对输入特征进行加权
        x = self.conv(x)        # [B, hidden_channels, H, W]
        x = self.relu(x)
        x = self.pool(x)        # [B, hidden_channels, 1, 1]
        x = torch.flatten(x, 1) # [B, hidden_channels]
        x = self.dropout(x)
        x = self.fc(x)          # [B, num_classes]
        return x





class GuidedWeightedCatFusion(nn.Module):
    def __init__(self, channels0_1=216, channels2=176, channels3=96, channels_mask=30, fused_channels=216, num_classes: int = 1, hidden_channels: int = None):
        super(GuidedWeightedCatFusion, self).__init__()

        # 可学习缩放因子
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w3 = nn.Parameter(torch.tensor(1.0))

        # 通道拼接后为 216+216+176+96 = 704
        self.compress = nn.Sequential(
            nn.Conv2d(704, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True)
        )
        
        self.compuse0 = nn.Conv2d(channels_mask, channels0_1, kernel_size=1)
        self.compuse1 = nn.Conv2d(channels_mask, channels0_1, kernel_size=1)
        self.compuse2 = nn.Conv2d(channels_mask, channels2, kernel_size=1)
        self.compuse3 = nn.Conv2d(channels_mask, channels3, kernel_size=1)
        
        if hidden_channels is None:
            hidden_channels = max(128, fused_channels // 2)  # 保证不小于32
        
        self.conv = nn.Conv2d(fused_channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, num_classes)

    # 针对每个特征大小插值掩码并引导增强
    def guide_feature(self, feat, mask):
            attn = F.interpolate(mask, size=feat.shape[2:], mode='bilinear', align_corners=True)  # [B, 30, H, W]
            
            return feat * attn  # 引导增强

    def forward(self, class_inp0, class_inp1, class_inp2, class_inp3, mask_out):
        # mask_out      torch.Size([24, 30, 128, 128])
        # class_inp3    torch.Size([24, 96, 64, 64])
        # class_inp2    torch.Size([24, 176, 32, 32])
        # class_inp1    torch.Size([24, 216, 16, 16])
        # class_inp0    torch.Size([24, 216, 16, 16]) fusion feature
        
        # 在原始分辨率上引导增强
        mask_out0 = self.compuse0(mask_out)
        class_inp0 = self.guide_feature(class_inp0, mask_out0)
        
        mask_out1 = self.compuse1(mask_out)
        class_inp1 = self.guide_feature(class_inp1, mask_out1)
        
        mask_out2 = self.compuse2(mask_out)
        class_inp2 = self.guide_feature(class_inp2, mask_out2)
        
        mask_out3 = self.compuse3(mask_out)
        class_inp3 = self.guide_feature(class_inp3, mask_out3)

        # 插值至统一尺寸
        target_size = class_inp0.shape[2:]
        class_inp0 = F.interpolate(class_inp0, size=target_size, mode='bilinear', align_corners=True)
        class_inp1 = F.interpolate(class_inp1, size=target_size, mode='bilinear', align_corners=True)
        class_inp2 = F.interpolate(class_inp2, size=target_size, mode='bilinear', align_corners=True)
        class_inp3 = F.interpolate(class_inp3, size=target_size, mode='bilinear', align_corners=True)

        # 拼接融合（带缩放因子）
        fused = torch.cat([
            self.w0 * class_inp0,
            self.w1 * class_inp1,
            self.w2 * class_inp2,
            self.w3 * class_inp3
        ], dim=1)  # shape: [B, 704, 16, 16]

        fused = self.compress(fused)
        fused = self.conv(fused)        # [B, hidden_channels, H, W]
        fused = self.relu(fused)
        fused = self.pool(fused)        # [B, hidden_channels, 1, 1]
        fused = torch.flatten(fused, 1) # [B, hidden_channels]
        fused = self.fc(fused)          # [B, num_classes]
        
        return fused


class Class_head(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, drop=0.1):
        super(Class_head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channels, num_classes) if num_classes > 0 else nn.Identity()
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.head(x)
        return x
    

class CMENet(nn.Module):
    def __init__(self, config, backbone='CME', drop=0.1):
        super(CMENet, self).__init__()
        assert backbone in ['CME']

        if backbone == 'CME':
            self.backbone1 = CME(img_size=128, in_chans=30)
            self.backbone2 = CME(img_size=128, in_chans=30)
            # self.backbone3 = CME(img_size=128, in_chans=30)
            self.backbone4 = CME(img_size=128, in_chans=30)
        else:
            raise NotImplementedError

        self.head = _FCNHead(96, 30, drop=drop)

        # t1, t1c, t2, tof
        """这里期望这四个损失函数能够调整一下各自优化一下自己的特征提取"""
        
        self.classifier_fusion = Class_head(in_channels=216, num_classes=config.num_classes)
        
        # self.classifier_t1 = ClassifierHead(in_channels=216, mask_channels=30, num_classes=config.num_classes)
        # self.classifier_t1c = ClassifierHead(in_channels=216, mask_channels=30, num_classes=config.num_classes)
        # self.classifier_tof = ClassifierHead(in_channels=216, mask_channels=30, num_classes=config.num_classes)
        # self.classifier_fusion = ClassifierHead(in_channels=216, mask_channels=30, num_classes=config.num_classes)
            
        self.ccaff3 = CCAFF_block(channels=216, out_channels=216, r=8)
        self.ccaff2 = CCAFF_block(channels=216, out_channels=176)
        self.ccaff1 = CCAFF_block(channels=176, out_channels=96)

        self.num_images = 0

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_segmentation_features(self, t1, stage="seg"):
        _, _, hei, wid = t1.shape
        # 骨干网络获取不同尺度的特征
        out1 = self.backbone1(t1)
        c1, c2, c3, c4 = out1

        totalout = self.ccaff3(c4, c3)  # 跨尺度融合 (4层,3层)
        
        totalout = F.interpolate(totalout, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        totalout = self.ccaff2(totalout, c2)  # 跨尺度融合 (3层,2层)
        
        totalout = F.interpolate(totalout, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        totalout = self.ccaff1(totalout, c1)  # 跨尺度融合 (2层,1层)
        
        pred = self.head(totalout)  # 预测分割结果
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)
        mask_out = self.sigmoid(out)  # 输出分割掩码
        
        return c4, mask_out
        

    def forward(self, t1, stage="seg", img_name="None"):

        """如果stage="cls"那么就冻结这部分代码，一直到mask_out = self.sigmoid(out)  # 输出分割掩码"""
        
        # c4            torch.Size([24, 216, 16, 16])
        # a4            torch.Size([24, 216, 16, 16])
        # d4            torch.Size([24, 216, 16, 16])
        # e4            torch.Size([24, 216, 16, 16])
        # mask_out      torch.Size([24, 30, 128, 128])
        # class_inp3    torch.Size([24, 96, 64, 64])
        # class_inp2    torch.Size([24, 176, 32, 32])
        # class_inp1    torch.Size([24, 216, 16, 16])
        # class_inp0    torch.Size([24, 216, 16, 16])
        
        feature, mask_out = self._get_segmentation_features(t1)
        
        if stage == "seg":  # 分割阶段 只需要分割掩码
            
            dummy_class = torch.ones((mask_out.shape[0], 1), device=mask_out.device)
            
            t1_class = dummy_class
            t1c_class = dummy_class
            tof_class = dummy_class
            fusion_class = dummy_class
            
            return mask_out, t1_class, t1c_class, tof_class, fusion_class
        
        elif stage == "cls":  # 分类阶段    
            # 融合分类结果
            fusion_class = self.classifier_fusion(feature)
            t1_class = fusion_class
            t1c_class = fusion_class
            tof_class = fusion_class
            return mask_out, t1_class, t1c_class, tof_class, fusion_class
        else:
            raise ValueError(f"Unsupported stage: {stage}")
