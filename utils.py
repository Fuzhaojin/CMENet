# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import nibabel as nib
import pandas as pd


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )



def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



# def save_imgs(img, msk, msk_pred, i, save_path, threshold=0.5, test_data_name=None):
#     img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
#     img = img / 255. if img.max() > 1.1 else img
#
#     msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
#
#     # fig = plt.figure(figsize=(2.56, 2.56))
#     # fig.subplots_adjust(left=0, right=1, bottom=0, top=0)
#     plt.imshow(img, cmap='gray')
#     plt.axis('off')
#
#     if test_data_name is not None:
#         save_path = save_path + test_data_name + '_'
#
#     plt.savefig(save_path + str(i) + '.png', bbox_inches='tight', pad_inches=0)
#     plt.close()

def save_imgs(img, msk, msk_pred, i, save_path, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img

    msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    fig, ax = plt.subplots(figsize=(256/72, 256/72))
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path + str(i) + '.png', dpi=72)
    plt.close()


def save_nii_msk(msk, msk_pred, i, name, save_path, threshold=0.5, test_data_name=None):
    # 处理 msk_pred 数据
    msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
    msk_pred = msk_pred.astype(np.int16)  # 将数据类型转换为 int16

    # 保存 msk_pred 为 NIfTI 文件
    msk_pred_nii = nib.Nifti1Image(msk_pred, np.eye(4))  # 使用单位矩阵作为 affine
    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    nib.save(msk_pred_nii, save_path + str(name[0])+ "_label" + '.nii')  # 保存为 .nii 格式

    # 保存 msk 的图像
    fig, ax = plt.subplots(figsize=(256 / 72, 256 / 72))
    ax.imshow(np.squeeze(msk)[0], cmap='gray')  # 显示第一层的图像
    ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path + str(name[0]) + "_label" + '.png', dpi=72)
    plt.close()
    
# import pdb
# def save_nii_mskpred(msk_pred, i, name, save_path, threshold=0.5, test_data_name=None):

#     # 处理 msk_pred 数据
#     pdb.set_trace()
#     msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
#     msk_pred = msk_pred.astype(np.int16)  # 将数据类型转换为 int16

#     # 保存 msk_pred 为 NIfTI 文件
#     msk_pred_nii = nib.Nifti1Image(msk_pred, np.eye(4))  # 使用单位矩阵作为 affine
#     if test_data_name is not None:
#         save_path = save_path + test_data_name + '_'
#     nib.save(msk_pred_nii, save_path + str(name[0]) + "_pred" + '.nii')  # 保存为 .nii 格式

#     # 保存 msk_pred 的图像
#     fig, ax = plt.subplots(figsize=(256 / 72, 256 / 72))
#     ax.imshow(msk_pred[0], cmap='gray')  # 显示第一层的预测结果
#     ax.axis('off')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.savefig(save_path + str(name[0]) + '_pred.png', dpi=72)
#     plt.close()
def save_nii_mskpred(msk_pred, i, name, save_path, threshold=0.5, test_data_name=None):
    # 确保目录存在
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    
    # 处理 msk_pred 数据
    msk_pred = np.squeeze(msk_pred, axis=0)  # 去掉 batch 维度 -> (30, 128, 128)
    msk_pred = np.where(msk_pred > threshold, 1, 0)
    
    # 调整维度顺序：将第0轴（30）移动到末尾 -> (128, 128, 30)
    msk_pred = np.transpose(msk_pred, (1, 2, 0))  
    msk_pred = msk_pred.astype(np.int16)

    # 保存 NIfTI 文件（维度为 128x128x30）
    msk_pred_nii = nib.Nifti1Image(msk_pred, affine=np.eye(4))
    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    nib.save(msk_pred_nii, save_path + str(name[0]) + "_pred.nii")

    # 保存预览图像（取中间切片）
    fig, ax = plt.subplots(figsize=(256/72, 256/72))
    mid_slice = msk_pred.shape[-1] // 2  # 取中间位置的切片
    ax.imshow(msk_pred[..., mid_slice], cmap='gray')  # 显示中间切片
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path + str(name[0]) + '_pred.png', dpi=72)
    plt.close()

# def save_class_out(msk_pred, clabel, i, name, file_name, save_path, threshold=0.5, test_data_name=None):
#     """
#     保存分类结果到CSV
#     参数说明：
#     - cls_out: 模型输出的 logits [batch_size, 1]
#     - clabel: 真实标签 [batch_size, ]
#     - batch_idx: 当前批次索引（用于生成唯一文件名）
#     - names: 样本名称列表 [batch_size, ]
#     - save_path: 保存路径
#     - threshold: 分类阈值
#     - test_data_name: 数据集标识（用于文件名）
#     """
#     # 确保目录存在
#     print(save_path)
#     os.makedirs(save_path, exist_ok=True)
#     filepath = os.path.join(save_path, file_name)

#     batch_df = pd.DataFrame({
#         'name': name,
#         'class_label': clabel,
#         'class_out': msk_pred,
#     })

#     # 生成文件名（包含批次和数据集信息）
#     # filename = f"{str(name[0])}_class.csv"
#     # 检查文件是否存在，决定写入模式
#     file_exists = os.path.exists(filepath)
#     mode = 'a' if file_exists else 'w'
#     header = not file_exists  # 仅首次写入时包含表头

#     batch_df.to_csv(filepath, mode=mode, header=header, index=False)

def save_class_out(msk_pred, clabel, i, name, file_name, save_path, threshold=0.5, test_data_name=None, y_prob=None):
    """
    保存分类结果到CSV
    参数说明：
    - cls_out: 模型输出的 logits [batch_size, 1]
    - clabel: 真实标签 [batch_size, ]
    - batch_idx: 当前批次索引（用于生成唯一文件名）
    - names: 样本名称列表 [batch_size, ]
    - save_path: 保存路径
    - threshold: 分类阈值
    - test_data_name: 数据集标识（用于文件名）
    - y_prob: 预测概率值
    """
    # 确保目录存在
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, file_name)

    data_dict = {
        'name': name,
        'class_label': clabel,
        'class_out': msk_pred,
    }
    
    # 如果提供了概率值，则添加到DataFrame中
    if y_prob is not None:
        data_dict['probability'] = y_prob
    
    batch_df = pd.DataFrame(data_dict)

    # 生成文件名（包含批次和数据集信息）
    # filename = f"{str(name[0])}_class.csv"
    # 检查文件是否存在，决定写入模式
    file_exists = os.path.exists(filepath)
    mode = 'a' if file_exists else 'w'
    header = not file_exists  # 仅首次写入时包含表头

    batch_df.to_csv(filepath, mode=mode, header=header, index=False)

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss



class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        t1, t1c, t2, tof, mask, seg_result = data

        # 将数据转换为PyTorch张量
        t1 = torch.tensor(t1, dtype=torch.float32)
        t1c = torch.tensor(t1c, dtype=torch.float32)
        t2 = torch.tensor(t2, dtype=torch.float32)
        tof = torch.tensor(tof, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        seg_result = torch.tensor(seg_result, dtype=torch.float32)
        

        t1 = t1.permute(2, 0, 1)
        t1c = t1c.permute(2, 0, 1)
        t2 = t2.permute(2, 0, 1)
        tof = tof.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        seg_result = seg_result.permute(2, 0, 1)

        return t1, t1c, t2, tof, mask, seg_result
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        t1, t1c, t2, tof, mask, seg_result = data

        t1 = TF.resize(t1, [self.size_h, self.size_w])
        t1c = TF.resize(t1c, [self.size_h, self.size_w])
        t2 = TF.resize(t2, [self.size_h, self.size_w])
        tof = TF.resize(tof, [self.size_h, self.size_w])
        mask = TF.resize(mask, [self.size_h, self.size_w])
        seg_result = TF.resize(seg_result, [self.size_h, self.size_w])

        return t1, t1c, t2, tof, mask, seg_result
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        t1, t1c, t2, tof, mask, seg_result = data
        if random.random() < self.p: 
            return TF.hflip(t1), TF.hflip(t1c), TF.hflip(t2), TF.hflip(tof), TF.hflip(mask), TF.hflip(seg_result)
        else: 
            return t1, t1c, t2, tof, mask, seg_result
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        t1, t1c, t2, tof, mask, seg_result = data
        if random.random() < self.p: 
            return TF.vflip(t1), TF.vflip(t1c), TF.vflip(t2), TF.vflip(tof), TF.vflip(mask), TF.vflip(seg_result)
        else: 
            return t1, t1c, t2, tof, mask, seg_result


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        t1, t1c, t2, tof, mask, seg_result = data
        if random.random() < self.p: 
            return TF.rotate(t1,self.angle), TF.rotate(t1c,self.angle), TF.rotate(t2,self.angle), TF.rotate(tof,self.angle), TF.rotate(mask,self.angle), TF.rotate(seg_result,self.angle)
        else: 
            return t1, t1c, t2, tof, mask, seg_result


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'caro' or data_name == 'XW_asymptomatic' or data_name == 'XW_symptomatic' or data_name == 'XW_mild' or data_name == 'XW_severe':
            if train:
                self.t1_mean = 725.0311808419307
                self.t1_std = 414.52469564960836

                self.t1c_mean = 1025.7849748164513
                self.t1c_std = 582.4112496216736
                
                self.t2_mean = 474.4876157225555
                self.t2_std = 331.91799508189393
                
                self.tof_mean = 90.21018866611392
                self.tof_std = 63.18413996106856
            else:
                self.t1_mean = 708.553682879672
                self.t1_std = 406.7009108433941

                self.t1c_mean = 1001.2429825412771
                self.t1c_std = 580.6942490142748
                
                self.t2_mean = 467.25204145921384
                self.t2_std = 327.37961161368565
                
                self.tof_mean = 89.5323895397082
                self.tof_std = 63.63661500556477
        if data_name == 'new_caro' or data_name == 'final':
            if train:
                self.t1_mean = 228.4063241394839
                self.t1_std = 219.4507714371974

                self.t1c_mean = 286.16820705066255
                self.t1c_std = 266.76549874544384

                self.t2_mean = 99.0702900298062
                self.t2_std = 107.80097140053654

                self.tof_mean = 120.00167597842182
                self.tof_std = 169.91381313849973
            else:
                self.t1_mean = 228.4063241394839
                self.t1_std = 219.4507714371974

                self.t1c_mean = 286.16820705066255
                self.t1c_std = 266.76549874544384

                self.t2_mean = 99.0702900298062
                self.t2_std = 107.80097140053654

                self.tof_mean = 120.00167597842182
                self.tof_std = 169.91381313849973
        
        if data_name == 'shengli' or data_name == 'SL_asymptomatic' or data_name == 'SL_symptomatic' or data_name == 'SL_mild' or data_name == 'SL_severe':
            self.t1_mean = 403.31957312137416
            self.t1_std = 351.208508650003
            
            self.t2_mean = 403.31957312137416
            self.t2_std = 351.208508650003
            
            self.t1c_mean = 520.5141782265769
            self.t1c_std = 528.7899772081413
            
            self.tof_mean = 158.10744609183283
            self.tof_std = 170.68349007224714
        
        if data_name == 'crop30':
            self.t1_mean = 794.3755463214322
            self.t1_std = 435.97570244193116
            
            self.t2_mean = 794.3755463214322
            self.t2_std = 435.97570244193116
            
            self.t1c_mean = 1134.2820776928634
            self.t1c_std = 634.7785068333952
            
            self.tof_mean = 93.73909676172131
            self.tof_std = 65.08780647313104
        
        if data_name == 'weifang':
            self.t1_mean = 113.46602452004603
            self.t1_std = 150.94920970682367
            
            self.t2_mean = 113.46602452004603
            self.t2_std = 150.94920970682367
            
            self.t1c_mean = 189.61172412014747
            self.t1c_std = 254.62022664261187
            
            self.tof_mean = 142.5684370331678
            self.tof_std = 240.02112241334328
        if data_name == 'shenzhen' or data_name == 'SZ_asymptomatic' or data_name == 'SZ_symptomatic' or data_name == 'SZ_mild' or data_name == 'SZ_severe':
            self.t1_mean = 659.5759661456812
            self.t1_std = 324.4760648388018
            
            self.t2_mean = 659.5759661456812
            self.t2_std = 324.4760648388018
            
            self.t1c_mean = 752.0163248411257
            self.t1c_std = 486.17903273906694
            
            self.tof_mean = 67.08690045152915
            self.tof_std = 52.061420139010444
            
        if data_name == 'QZ_asymptomatic' or data_name == 'QZ_symptomatic' or data_name == 'QZ_mild' or data_name == 'QZ_severe':
            self.t1_mean = 781.6540089665082
            self.t1_std = 419.95287392329857
            
            self.t2_mean = 781.6540089665082
            self.t2_std = 419.95287392329857
            
            self.t1c_mean = 1133.8564001944492
            self.t1c_std = 627.0301772418015
            
            self.tof_mean = 94.05992326964723
            self.tof_std = 64.81216791477154
            

    def __call__(self, data):
        t1, t1c, t2, tof, mask, seg_result = data

        # 处理 t1 通道
        t1_normalized = (t1 - self.t1_mean) / self.t1_std
        t1_min, t1_max = t1_normalized.min(), t1_normalized.max()
        t1_normalized = ((t1_normalized - t1_min) / (t1_max - t1_min)) * 255.

        # 处理 t1c 通道
        t1c_normalized = (t1c - self.t1c_mean) / self.t1c_std
        t1c_min, t1c_max = t1c_normalized.min(), t1c_normalized.max()
        t1c_normalized = ((t1c_normalized - t1c_min) / (t1c_max - t1c_min)) * 255.

        # 处理 t2 通道
        t2_normalized = (t2 - self.t2_mean) / self.t2_std
        t2_min, t2_max = t2_normalized.min(), t2_normalized.max()
        t2_normalized = ((t2_normalized - t2_min) / (t2_max - t2_min)) * 255.

        # 处理 tof 通道
        tof_normalized = (tof - self.tof_mean) / self.tof_std
        tof_min, tof_max = tof_normalized.min(), tof_normalized.max()
        tof_normalized = ((tof_normalized - tof_min) / (tof_max - tof_min)) * 255.

        return t1_normalized, t1c_normalized, t2_normalized, tof_normalized, mask, seg_result

