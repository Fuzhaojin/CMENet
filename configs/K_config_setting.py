# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project

from torchvision import transforms
from utils import *
from Loss import *
from datetime import datetime

class setting_config:
    """
    the config of training setting.
    
    分类任务300
    分割任务300
    ==============================================
    network         workname            workname              说明
         
    CMENet          network4_NK (w/o seg)  network4_NK_seg     这是分割版本模型
    final_class     final_class_cls                         
    final_class2    final_class2_cls                       
    
    final_use_seg   final_use_seg_cls       这个版本是最终的分类模型, 使用分割结果作为输入
    
    ===============对比方法========================
    UNet            UNet_cls            UNet_seg             
    AttUNet         AttUNet_cls         AttUNet_seg
    CASF            CASF_cls            CASF_seg
    NestedUNet      NestedUNet_cls      NestedUNet_seg
    UNetv2          UNetv2_cls          UNetv2_seg
    nnUnet                              nnUnet_seg
    
    =============分类新的对比方法, 整理在A_NewCompare_methods.py中===========
    # 需要在CARO_datasets_Xlsx中更改对应的seg_result路径
    
    ResNet50        ResNet50_cls       
    ResNet101       ResNet101_cls
    Densenet121     Densenet121_cls
    Vit             Vit_cls
    Efficientnet_b3 Efficientnet_b3_cls
    
    
    =============消融实验==========================
    
    CMENet          network4_NK_cls                     这个版本作为骨干网络提取特征之后进行分类的消融实验
    xr_t1           xr_t1_cls           xr_t1_seg
    xr_t1c          xr_t1c_cls          xr_t1c_seg 
    xr_tof          xr_tof_cls          xr_tof_seg
    share_encoder   share_encoder_cls   share_encoder_seg
    final_loss      final_loss_cls      final_loss_seg
    no_ccaff        no_ccaff_cls        no_ccaff_seg
    xr_transformer  xr_transformer_cls  xr_transformer_seg
    xr_cnn          xr_cnn_cls          xr_cnn_seg
    """
    network = 'CMENet'
    
    """训练关键信息"""
    stage = 'seg'  # seg or cls
    workname = "network4_NK_seg" # 训练名称  # network4_noT2_2NoKtrian2
    epochs = 300  # 训练轮数
    batch_size = 24
    val_interval = 1  # 没多少轮验证一次
    save_interval = 1  # 多少轮保存一次
    device_ids = [2]
    
    dloss_weight32 = 0.3
    dloss_weight64 = 0.5

    test_weights = ''
    
    """
    QZ_asymptomatic  QZ_symptomatic  QZ_mild  QZ_severe
    
    SZ_asymptomatic  SZ_symptomatic  SZ_mild  SZ_severe
    
    SL_asymptomatic  SL_symptomatic  SL_mild  SL_severe
    
    XW_asymptomatic  XW_symptomatic  XW_mild  XW_severe
    
    """
    datasets = 'caro'
    
    if datasets == 'caro':
        vis_file = ''
        xlsx_dir = './data/CarotidData/class_label_new2.xlsx'
        txt_file = './data/CarotidData/train_new2.txt'  # './data/CarotidData/selected_T1_2.txt'
        test_txt_file = './data/CarotidData/test_new2.txt'
        vis_txt_file = './data/CarotidData/visual_new2.txt'
        images_dir = './data/CarotidData/images'
        # masks_dir = './data/CarotidData/masks2'
        masks_dir = './data/CarotidData/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2

    # if datasets == 'caro':
    #         vis_file = ''
    #         xlsx_dir = './data/c/class_label.xlsx'
    #         txt_file = './data/c/test.txt'  # './data/CarotidData/selected_T1_2.txt'
    #         test_txt_file = './data/c/test.txt'
    #         vis_txt_file = './data/c/test.txt'
    #         images_dir = './data/c/images'
    #         # masks_dir = './data/CarotidData/masks2'
    #         masks_dir = './data/c/masks'
    #         input_size_h = 128
    #         input_size_w = 128
    #         input_channels = 30
    #         out_channels = 30
    #         num_classes = 2
        
    # final数据集为外部验证数据集，无mask
    elif datasets == 'final':
        xlsx_dir = './data/final/class_label.xlsx'
        txt_file = './data/final/test.txt'  # './data/CarotidData/selected_T1_2.txt'
        test_txt_file = './data/final/test.txt'
        images_dir = './data/final/images'
        masks_dir = './data/final/images'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2 # 根据你的数据集类别数量进行调整
    
    elif datasets == 'crop30':
        vis_file = ''
        xlsx_dir = './data/crop30/class_label.xlsx'
        txt_file = './data/crop30/test.txt'  
        test_txt_file = './data/crop30/test.txt'
        # vis_txt_file = './data/crop30/visual_new2.txt'
        images_dir = './data/crop30/images'
        masks_dir = './data/crop30/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
        
    elif datasets == 'shengli':
        vis_file = ''
        xlsx_dir = './data/shengli/class_label.xlsx'
        txt_file = './data/shengli/test.txt'  
        test_txt_file = './data/shengli/test.txt'
        # vis_txt_file = './data/shengli/visual_new2.txt'
        images_dir = './data/shengli/images'
        masks_dir = './data/shengli/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    
    elif datasets == 'shenzhen':
        vis_file = ''
        xlsx_dir = './data/SZ/shenzhen/combined label.xlsx'
        txt_file = './data/SZ/shenzhen/test.txt'  
        test_txt_file = './data/SZ/shenzhen/test.txt'
        # vis_txt_file = './data/crop30/visual_new2.txt'
        images_dir = './data/SZ/shenzhen/images'
        masks_dir = './data/SZ/shenzhen/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    
    # 亚组
    # 前瞻亚组
    elif datasets == 'QZ_asymptomatic':
        vis_file = ''
        xlsx_dir = './data/QZ/asymptomatic/asymptomatic.xlsx'
        txt_file = './data/QZ/asymptomatic/test.txt'  
        test_txt_file = './data/QZ/asymptomatic/test.txt'
        # vis_txt_file = './data/crop30/visual_new2.txt'
        images_dir = './data/QZ/asymptomatic/images'
        masks_dir = './data/QZ/asymptomatic/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'QZ_symptomatic':
        vis_file = ''
        xlsx_dir = './data/QZ/symptomatic/symptomatic.xlsx'
        txt_file = './data/QZ/symptomatic/test.txt'  
        test_txt_file = './data/QZ/symptomatic/test.txt'
        # vis_txt_file = './data/crop30/visual_new2.txt'
        images_dir = './data/QZ/symptomatic/images'
        masks_dir = './data/QZ/symptomatic/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'QZ_mild':
        vis_file = ''
        xlsx_dir = './data/QZ/mild to moderate/mild to moderate.xlsx'
        txt_file = './data/QZ/mild to moderate/test.txt'  
        test_txt_file = './data/QZ/mild to moderate/test.txt'
        # vis_txt_file = './data/crop30/visual_new2.txt'
        images_dir = './data/QZ/mild to moderate/images'
        masks_dir = './data/QZ/mild to moderate/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'QZ_severe':
        vis_file = ''
        xlsx_dir = './data/QZ/severe/severe.xlsx'
        txt_file = './data/QZ/severe/test.txt'  
        test_txt_file = './data/QZ/severe/test.txt'
        # vis_txt_file = './data/crop30/visual_new2.txt'
        images_dir = './data/QZ/severe/images'
        masks_dir = './data/QZ/severe/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    
    # 深圳亚组
    elif datasets == 'SZ_asymptomatic':
        vis_file = ''
        xlsx_dir = './data/SZ/asymptomatic/asymptomatic.xlsx'
        txt_file = './data/SZ/asymptomatic/test.txt'  
        test_txt_file = './data/SZ/asymptomatic/test.txt'
        images_dir = './data/SZ/shenzhen/images'
        masks_dir = './data/SZ/shenzhen/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'SZ_symptomatic':
        vis_file = ''
        xlsx_dir = './data/SZ/symptomatic/symptomatic.xlsx'
        txt_file = './data/SZ/symptomatic/test.txt'  
        test_txt_file = './data/SZ/symptomatic/test.txt'
        images_dir = './data/SZ/shenzhen/images'
        masks_dir = './data/SZ/shenzhen/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'SZ_mild':
        vis_file = ''
        xlsx_dir = './data/SZ/mild to moderate/mild to moderate.xlsx'
        txt_file = './data/SZ/mild to moderate/test.txt'  
        test_txt_file = './data/SZ/mild to moderate/test.txt'
        images_dir = './data/SZ/shenzhen/images'
        masks_dir = './data/SZ/shenzhen/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'SZ_severe':
        vis_file = ''
        xlsx_dir = './data/SZ/severe/severe.xlsx'
        txt_file = './data/SZ/severe/test.txt'  
        test_txt_file = './data/SZ/severe/test.txt'
        images_dir = './data/SZ/shenzhen/images'
        masks_dir = './data/SZ/shenzhen/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    
    # 省立亚组
    elif datasets == 'SL_asymptomatic':
        vis_file = ''
        xlsx_dir = './data/SL/asymptomatic/asymptomatic.xlsx'
        txt_file = './data/SL/asymptomatic/test.txt'  
        test_txt_file = './data/SL/asymptomatic/test.txt'
        images_dir = './data/SL/asymptomatic/images'
        masks_dir = './data/SL/asymptomatic/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'SL_symptomatic':
        vis_file = ''
        xlsx_dir = './data/SL/symptomatic/symptomatic.xlsx'
        txt_file = './data/SL/symptomatic/test.txt'  
        test_txt_file = './data/SL/symptomatic/test.txt'
        images_dir = './data/SL/symptomatic/images'
        masks_dir = './data/SL/symptomatic/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'SL_mild':
        vis_file = ''
        xlsx_dir = './data/SL/mild to moderate/mild to moderate.xlsx'
        txt_file = './data/SL/mild to moderate/test.txt'  
        test_txt_file = './data/SL/mild to moderate/test.txt'
        images_dir = './data/SL/mild to moderate/images'
        masks_dir = './data/SL/mild to moderate/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'SL_severe':
        vis_file = ''
        xlsx_dir = './data/SL/severe/severe.xlsx'
        txt_file = './data/SL/severe/test.txt'  
        test_txt_file = './data/SL/severe/test.txt'
        images_dir = './data/SL/severe/images'
        masks_dir = './data/SL/severe/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    
    # 宣武亚组
    elif datasets == 'XW_asymptomatic':
        vis_file = ''
        xlsx_dir = './data/XW/asymptomatic/asymptomatic.xlsx'
        txt_file = './data/XW/asymptomatic/test.txt'  
        test_txt_file = './data/XW/asymptomatic/test.txt'
        images_dir = './data/XW/asymptomatic/images'
        masks_dir = './data/XW/asymptomatic/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'XW_symptomatic':
        vis_file = ''
        xlsx_dir = './data/XW/symptomatic/symptomatic.xlsx'
        txt_file = './data/XW/symptomatic/test.txt'  
        test_txt_file = './data/XW/symptomatic/test.txt'
        images_dir = './data/XW/symptomatic/images'
        masks_dir = './data/XW/symptomatic/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'XW_mild':
        vis_file = ''
        xlsx_dir = './data/XW/mild to moderate/mild to moderate.xlsx'
        txt_file = './data/XW/mild to moderate/test.txt'  
        test_txt_file = './data/XW/mild to moderate/test.txt'
        images_dir = './data/XW/mild to moderate/images'
        masks_dir = './data/XW/mild to moderate/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    elif datasets == 'XW_severe':
        vis_file = ''
        xlsx_dir = './data/XW/severe/severe.xlsx'
        txt_file = './data/XW/severe/test.txt'  
        test_txt_file = './data/XW/severe/test.txt'
        images_dir = './data/XW/severe/images'
        masks_dir = './data/XW/severe/masks'
        input_size_h = 128
        input_size_w = 128
        input_channels = 30
        out_channels = 30
        num_classes = 2
    
    else:
        raise Exception('datasets in not right!')

    # k_fold_cross_validation
    k = 5
    seg_or_class = 200 

    criterion = CombinedLoss()
    criterion_class = ClassLoss()  # ClassCombinedLoss  ClassLoss() 
    # criterion = BceDiceLoss()
    # criterion = BCELoss()
    
    
    file_name = "total_class.csv"


    distributed = False
    local_rank = -1
    num_workers = 8
    seed = 42
    world_size = None
    rank = None
    amp = False
    

    # work_dir = 'results/' + network + '_NoK_' + datasets + 'stage_train' + '/'
    work_dir = 'results/' + network + '_NoK_' + datasets + '_' +workname + '/'
    # work_dir = 'results/' + network + '_NoK_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20  #

    threshold = 0.5

    train_transform = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
        ])
    test_transform = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'Adam'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001 # default: 1e-3 – learning rate # 5e-4
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.9 # default: 0 – momentum factor 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20