# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project

import os
import sys
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from dataset.caro_data import CARO_datasets, CustomTransform, CARO_datasets_Xlsx
from dataset.npy_datasets import NPY_datasets
from utils import *
from configs.K_config_setting import setting_config
import warnings
warnings.filterwarnings("ignore")
# from models.UNet import *


from models.FAT_Net import FAT_Net
from engine import *
# import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8'

# 加载数据
def load_data(config, xlsx_dir, test_file, txt_file, images_dir, masks_dir, batch_size):
    
    if config.datasets == 'caro':
        dataset = CARO_datasets_Xlsx(xlsx_dir=xlsx_dir, txt_file=txt_file, images_dir=images_dir, masks_dir=masks_dir, transform=config.train_transform)
        test_dataset = CARO_datasets_Xlsx(xlsx_dir=xlsx_dir, txt_file=test_file, images_dir=images_dir, masks_dir=masks_dir, transform=config.test_transform)
    else:
        dataset = NPY_datasets(config.data_path, config, train=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return test_dataset, dataset


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    # resume_model = os.path.join(checkpoint_dir, 'best-epoch1331-loss0.8262.pth')
    resume_model = os.path.join(checkpoint_dir, 'best_segmodel_for_class_train.pth')  # best_segmodel_for_class_train.pth
    original = os.path.join(config.work_dir, 'original')
    background = os.path.join(config.work_dir, 'background')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(original):
        os.makedirs(original)
    if not os.path.exists(background):
        os.makedirs(background)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)


    print('#----------Prepareing Models----------#')
    device_ids = config.device_ids
    device = torch.device('cuda:{}'.format(device_ids[0]) if torch.cuda.is_available() else 'cpu')
    print(device)
    # network = CarNet(in_channels=config.input_channels, num_classes=config.out_channels)
    # network = SLDDNet(backbone='SLDD')
    # network = Unet(in_ch=60, out_ch=60)
    
    if config.network == 'CMENet':
        from models.Model.main_models.network import CMENet
        network = CMENet(config=config)
    
    # 最终的分类模型。使用最后的结果进行分类
    elif config.network == 'final_class':
        from models.Model.main_models.network_final_class import CMENet
        network = CMENet(config=config)
    
    elif config.network == 'final_class2':
        from models.Model.main_models.network_final_class2 import CMENet
        network = CMENet(config=config)
    
    elif config.network == 'final_use_seg':
        from models.Model.main_models.network_final_class2_use_seg import CMENet
        network = CMENet(config=config)
        
    elif config.network == 'AttUNet':
        from models.compare_methods.Attunet_for_Car import AttU_Net
        network = AttU_Net(config=config,in_channel=30, out_channel=30)

    elif config.network == 'UNet':
        from models.compare_methods.UNet_for_Car import UNet
        network = UNet(config=config, in_ch=30, out_ch=30)

    elif config.network == 'CASF':
        from models.compare_methods.CASF.lib.TransFuse import TransFuse_S as CASF
        network = CASF(config=config)
    
    elif config.network == 'UNetv2':
        from models.compare_methods.unet_v2.UNet_v2 import UNetV2
        network = UNetV2(config=config, in_channels=30, out_channels=30)
    
    elif config.network == 'NestedUNet':
        from models.compare_methods.NestedUNet import NestedUNet
        network = NestedUNet(config=config)

    elif config.network == 'nnUnet':
        from models.compare_methods.nnUnet import UNet2DConfig, TriModalFusionUNet2D
        cfg = UNet2DConfig(
        in_channels=30,     # 这里对TriModal没用，但保留即可
        num_classes=30,      # 2类分割举例
        deep_supervision=False
        )
        network = TriModalFusionUNet2D(cfg, in_channels_per_modality=30)

    # 新的分类对比实验
    elif config.network == 'ResNet50':
        from models.compare_methods.A_NewCompare_methods import ResNet50_for_Car
        network = ResNet50_for_Car(config, in_channels=30, num_classes=config.num_classes)
        
    elif config.network == 'Densenet121':
        from models.compare_methods.A_NewCompare_methods import Densenet121_for_Car
        network = Densenet121_for_Car(config, in_channels=30, num_classes=config.num_classes)
    
    elif config.network == 'Efficientnet_b3':
        from models.compare_methods.A_NewCompare_methods import Efficientnet_b3_for_Car
        network = Efficientnet_b3_for_Car(config, in_channels=30, num_classes=config.num_classes)
    
    elif config.network == 'ResNet101':
        from models.compare_methods.A_NewCompare_methods import ResNet101_for_Car
        network = ResNet101_for_Car(config, in_channels=30, num_classes=config.num_classes)
    
    elif config.network == 'Vit':
        from models.compare_methods.A_NewCompare_methods import Vit_for_Car
        network = Vit_for_Car(config, in_channels=30, num_classes=config.num_classes)

    # 消融实验
    # 验证模态
    elif config.network == 'xr_t1' or config.network == 'xr_tof':
        from models.Model.main_models.network_xr_t1 import CMENet
        network = CMENet(config=config)
    
    elif config.network == 'xr_t1c':
        from models.Model.main_models.network_xr_t1c import CMENet
        network = CMENet(config=config)
    
    # 验证共享编码器
    elif config.network == 'share_encoder':
        from models.Model.main_models.network_share_encoder import CMENet
        network = CMENet(config=config)
    
    # 验证多损失    
    elif config.network == 'final_loss':
        from models.Model.main_models.network_final_loss import CMENet
        network = CMENet(config=config)
    
    # 验证骨干网络提取特征直接分类
    elif config.network == 'CMENet':
        from models.Model.main_models.network import CMENet
        network = CMENet(config=config)
        
    # 验证CCAFF融合
    elif config.network == 'no_ccaff':
        from models.Model.main_models.network_no_ccaff import CMENet
        network = CMENet(config=config)
    
    elif config.network == 'xr_transformer':
        from models.Model.main_models.network_xr_transformer import CMENet
        network = CMENet(config=config)
    
    elif config.network == 'xr_cnn':
        from models.Model.main_models.network_xr_cnn import CMENet
        network = CMENet(config=config)

    model = nn.DataParallel(network, device_ids=device_ids)
    model.to(device)


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    criterion_class = config.criterion_class
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()


    print('#----------Set other params----------#')
    min_loss = 999
    max_f1 = 0
    max_auroc_class = 0
    max_acc_class = 0

    start_epoch = 1
    min_epoch = 1

    # 加载预训练模型
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location = device)
        
        if config.stage == 'cls':
            # 分类阶段：只加载模型权重，其他参数重新开始
            saved_epoch = checkpoint['epoch']
            print("train class model using best seg model {}!".format(saved_epoch))
            model.load_state_dict(checkpoint['model_state_dict'])
            log_info = f'Loading segmentation model from {resume_model} for classification training. Starting from epoch 1.'
            logger.info(log_info)
        elif config.stage == 'seg':
            # 分割阶段：加载所有状态
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            saved_epoch = checkpoint['epoch']
            start_epoch = saved_epoch
            print("train seg model using best seg model {}!".format(saved_epoch))
            min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
            log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
            logger.info(log_info)
        else:
            raise ValueError(
                f"Invalid stage value: '{config.stage}'. "
                "Allowed values are 'seg' or 'cls'.")


    print('#----------Training----------#')
    # k_fold_cross_validation(config, config.k, config.batch_size, config.epochs, config.num_classes)
    
    test_dataset, dataset = load_data(config, config.xlsx_dir, config.test_txt_file, config.txt_file, config.images_dir, config.masks_dir, config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    

    for epoch in range(start_epoch, config.epochs + 1):

        train_one_epoch(
            train_loader,
            model,
            criterion,
            criterion_class,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            device,
            scaler=scaler,
        )
        if config.stage == 'seg':
            loss, f1 = val_one_epoch(
                val_loader,
                model,
                criterion,
                criterion_class,
                epoch,
                logger,
                config,
                device
            )
        elif config.stage == 'cls':
            loss, auroc_class, acc_class = val_one_epoch(
                val_loader,
                model,
                criterion,
                criterion_class,
                epoch,
                logger,
                config,
                device
            )

        """阶段选择"""
        # if epoch % config.save_interval == 0:
        # 分割最优模型
        if config.stage == 'seg':
            if f1 > max_f1 :
                # 用于test的最优模型，仅保存关键字典信息
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                # 用于分类训练的
                torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'best_segmodel_for_class_train.pth'))
                
                max_f1 = f1
                min_epoch = epoch
                log_info = 'best model save on {}!\n\n\n'.format(epoch)                
                print(log_info)
                logger.info(log_info)
        # 分类最优模型
        elif config.stage == 'cls':
            # if auroc_class > max_auroc_class:
            #     torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'class_best.pth'))
            #     max_auroc_class = auroc_class
            #     min_epoch = epoch
            #     log_info = f'class best model save !\n\n\n'
            #     print(log_info)
            #     logger.info(log_info)
                
            if acc_class > max_acc_class:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'class_best.pth'))
                max_acc_class = acc_class
                min_epoch = epoch
                log_info = f'class best model save !\n\n\n'
                print(log_info)
                logger.info(log_info)
        else:
            raise ValueError(
                f"Invalid stage value: '{config.stage}'. "
                "Allowed values are 'seg' or 'cls'.")

        if loss < min_loss:
            min_loss = loss

        torch.save(
        {
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))


    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device=torch.device('cuda:0')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=device)
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                criterion_class,
                logger,
                config,
                device
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )   


    if os.path.exists(os.path.join(checkpoint_dir, 'class_best.pth')):
        print('#----------Testing----------#')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device=torch.device('cuda:0')
        best_weight = torch.load(config.work_dir + 'checkpoints/class_best.pth', map_location=device)
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                criterion_class,
                logger,
                config,
                device
            )
        os.rename(
            os.path.join(checkpoint_dir, 'class_best.pth'),
            os.path.join(checkpoint_dir, f'class-best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )   


if __name__ == '__main__':
    config = setting_config
    main(config)