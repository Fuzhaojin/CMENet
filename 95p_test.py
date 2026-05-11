# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project

from configs.K_config_setting import setting_config
import torch
from torch import nn
# from models.UNet import *
from dataset.npy_datasets import NPY_datasets
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import torch
import os
from dataset.caro_data import CARO_datasets, CustomTransform, CARO_datasets_Xlsx
from sklearn.metrics import confusion_matrix
from utils import save_imgs,save_nii_msk, save_nii_mskpred, save_class_out
# from models.UNet_for_Car import Unet
# from models.attunet import AttU_Net
from models.Model.main_models.network import CMENet
# from models.CASF.lib.TransFuse import TransFuse_S as CASF
# from models.FAT_Net import FAT_Net
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import pdb

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
from sklearn.utils import resample

def load_data(config, xlsx_dir, test_file, txt_file, images_dir, masks_dir, batch_size):
    
    if config.datasets == 'caro':
        dataset = CARO_datasets_Xlsx(xlsx_dir=xlsx_dir, txt_file=txt_file, images_dir=images_dir, masks_dir=masks_dir, transform=config.train_transform)
        test_dataset = CARO_datasets_Xlsx(xlsx_dir=xlsx_dir, txt_file=test_file, images_dir=images_dir, masks_dir=masks_dir, transform=config.test_transform)
    elif config.datasets == 'new_caro' or config.datasets == 'final':
        dataset = CARO_datasets_Xlsx(xlsx_dir=xlsx_dir, txt_file=txt_file, images_dir=images_dir, masks_dir=masks_dir, transform=config.train_transform)
        test_dataset = CARO_datasets_Xlsx(xlsx_dir=xlsx_dir, txt_file=test_file, images_dir=images_dir, masks_dir=masks_dir, transform=config.test_transform)
    else:
        dataset = NPY_datasets(config.data_path, config, train=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return test_dataset, dataset



from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_cls_ci_bootstrap(y_true, y_pred, y_prob=None, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    accs, precs, recalls, f1s, aucs, specs = [], [], [], [], [], []

    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)))
        yt, yp = y_true[indices], y_pred[indices]
        accs.append(accuracy_score(yt, yp))
        precs.append(precision_score(yt, yp, zero_division=0))
        recalls.append(recall_score(yt, yp, zero_division=0))
        f1s.append(f1_score(yt, yp, zero_division=0))
        if y_prob is not None:
            aucs.append(roc_auc_score(yt, y_prob[indices]))
        # 👉 添加 specificity
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        TN, FP = cm[0, 0], cm[0, 1]
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specs.append(spec)

    def ci(arr):
        lower = np.percentile(arr, 2.5)
        upper = np.percentile(arr, 97.5)
        delta = (upper - lower) / 2
        return np.mean(arr), delta

    acc_mu, acc_ci = ci(accs)
    prec_mu, prec_ci = ci(precs)
    rec_mu, rec_ci = ci(recalls)
    f1_mu, f1_ci = ci(f1s)
    auc_mu, auc_ci = ci(aucs) if y_prob is not None else (None, None)
    spec_mu, spec_ci = ci(specs)

    return {
        'accuracy': (acc_mu, acc_ci),
        'precision': (prec_mu, prec_ci),
        'recall': (rec_mu, rec_ci),
        'f1': (f1_mu, f1_ci),
        'auroc': (auc_mu, auc_ci),
        'specificity': (spec_mu, spec_ci),
    }



def test_one_epoch(test_loader,
                    model,
                    criterion,
                    criterion_class,
                    config,
                    device,
                    result_path,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    sum_time = 0

    # 分类计数器
    class_preds = []
    clabels = []
    y_probs = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            # t1, t1c, t2, tof, targets, clabel, name = data
            t1, t1c, t2, tof, targets, clabel, seg_result, name = data
            
            t1 = t1.to(device, dtype=torch.float)
            t1c = t1c.to(device, dtype=torch.float)
            # t2 = t2.to(device, dtype=torch.float)
            tof = tof.to(device, dtype=torch.float)
            seg_result = seg_result.to(device, dtype=torch.float)
            
            targets = targets.to(device, dtype=torch.float)
            clabel = clabel.to(device, dtype=torch.long)
            
            # 启动计时器
            start_time = time.time()
            
            # 模型选择
            if config.network == "AttUNet" or config.network == "CASF" or config.network == "UNetv2" or config.network == 'NestedUNet':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, config.stage)

            elif config.network == "UNet" or config.network == "CMENet" or config.network == 'final_use_seg':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, seg_result, config.stage)
            
            if config.network == 'xr_t1':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, config.stage)
                
            elif config.network == 'xr_t1c':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, config.stage)
            
            elif config.network == 'xr_tof':
                out, t1_class, t1c_class, tof_class, cls_out = model(tof, config.stage)
            
            elif config.network == 'final_class' or config.network == 'final_class2' or config.network == 'share_encoder' or config.network == 'xr_transformer' or config.network == 'xr_cnn':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, config.stage)
            
            elif config.network == 'final_loss':
                out, cls_out = model(t1, t1c, tof, config.stage)
                t1_class = cls_out
                t1c_class = cls_out
                tof_class = cls_out
            
            elif config.network == 'no_ccaff':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, config.stage)
            
            
            if config.stage == 'seg':
                # 训练分割模型
                loss_mask = criterion(out, targets)
                loss = loss_mask
                             
            elif config.stage == 'cls':
                # 训练分类模型
                loss_t1 = criterion_class(t1_class, clabel)
                loss_t1c = criterion_class(t1c_class, clabel)
                # loss_t2 = criterion_class(t2_class, clabel) 
                loss_tof = criterion_class(tof_class, clabel)  
                loss_class = criterion_class(cls_out, clabel)
                loss = loss_class + loss_t1 + loss_t1c + loss_tof  

            
            # 停止计时器
            end_time = time.time()
            # 计算推理时间
            inference_time = end_time - start_time
            print(f"Inference Time: {inference_time} seconds")

            # 假设你已经有了模型的预测时间（秒）, 可以计算 FPS
            prediction_time = inference_time
            fps = 1 / prediction_time
            print(f"FPS: {fps}")

            sum_time = sum_time + inference_time
            print("sum", sum_time)

            if config.stage == 'seg':
                # 训练分割模型
                loss_mask = criterion(out, targets)
                loss = loss_mask
                # 分割结果汇总
                targets = targets.squeeze(1).cpu().detach().numpy()
                gts.append(targets)
                if type(out) is tuple:
                    out = out[0]
                out = out.squeeze(1).cpu().detach().numpy()
                preds.append(out)     
                
                save_nii_mskpred(out, i, name, result_path + 'outputs/', config.threshold, test_data_name=test_data_name)
                
            elif config.stage == 'cls':
                # 训练分类模型
                loss_class = criterion_class(cls_out, clabel)
                loss = loss_class
                
                # 获取概率和预测类别
                y_prob = torch.softmax(cls_out, dim=1)[:, 1]  # 二分类，取类别 1 的概率
                
                predict_y = (y_prob > 0.5).long()  # 阈值划分为 0 或 1
                # 分类结果汇总
                predict_y = predict_y.squeeze().cpu().detach().numpy()
                cls_out = cls_out.squeeze().cpu().detach().numpy()
                class_preds.append(predict_y)

                clabel = clabel.squeeze().cpu().detach().numpy()
                clabels.append(clabel)
                
                y_prob = y_prob.squeeze().cpu().detach().numpy()
                y_probs.append(y_prob)
                
                save_class_out(predict_y, clabel, i, name, config.file_name, result_path + 'class_outputs/', config.threshold, test_data_name=test_data_name, y_prob=y_prob)
                
            else:
                # 默认分割模型
                loss_mask = criterion(out, targets)
                loss = loss_mask 
            
            # 计算损失  
            loss_list.append(loss.item())
                    
        # 分割计算
        if config.stage == 'seg':
            preds = np.array(preds)  # shape: [N, H, W]
            gts = np.array(gts)      # shape: [N, H, W]
            y_preds = np.where(preds >= config.threshold, 1, 0)
            y_trues = np.where(gts >= 0.5, 1, 0)

            acc_list, prec_list, rec_list, f1_list, miou_list = [], [], [], [], []

            for i in range(len(y_preds)):
                pred_i = y_preds[i].flatten()
                true_i = y_trues[i].flatten()
                cm = confusion_matrix(true_i, pred_i, labels=[0, 1])
                TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

                acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
                prec = TP / (TP + FP) if (TP + FP) != 0 else 0
                rec = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
                iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

                acc_list.append(acc)
                prec_list.append(prec)
                rec_list.append(rec)
                f1_list.append(f1)
                miou_list.append(iou)

            # 均值与标准差
            acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
            prec_mean, prec_std = np.mean(prec_list), np.std(prec_list)
            rec_mean, rec_std = np.mean(rec_list), np.std(rec_list)
            f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
            miou_mean, miou_std = np.mean(miou_list), np.std(miou_list)

            if test_data_name is not None:
                print(f"test_datasets_name: {test_data_name}")

            print("Mask test of best model:")
            print(f"  loss: {np.mean(loss_list):.4f}")
            print(f"  accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
            print(f"  f1_or_dsc: {f1_mean:.4f} ± {f1_std:.4f}")
            print(f"  miou: {miou_mean:.4f} ± {miou_std:.4f}")
            print(f"  precision: {prec_mean:.4f} ± {prec_std:.4f}")
            print(f"  recall: {rec_mean:.4f} ± {rec_std:.4f}")
            

            log_info = ""
            log_info += "Mask test of best model:\n"
            log_info += f"  loss: {np.mean(loss_list):.4f}\n"
            log_info += f"  accuracy: {acc_mean:.4f} ± {acc_std:.4f}\n"
            log_info += f"  f1_or_dsc: {f1_mean:.4f} ± {f1_std:.4f}\n"
            log_info += f"  miou: {miou_mean:.4f} ± {miou_std:.4f}\n"
            log_info += f"  precision: {prec_mean:.4f} ± {prec_std:.4f}\n"
            log_info += f"  recall: {rec_mean:.4f} ± {rec_std:.4f}\n"
            

            with open(os.path.join(result_path, 'log', 'metrics_log.txt'), 'a') as f:
                f.write(log_info + '\n')


        # 分类计算
        if config.stage == 'cls':
            # 原始 torchmetrics 指标计算
            acc = Accuracy(task="binary").to('cpu')
            prec = Precision(task="binary").to('cpu')
            rec = Recall(task="binary").to('cpu')  # 即Sensitivity
            f1 = F1Score(task="binary").to('cpu')
            auroc = AUROC(task="binary").to('cpu')

            class_preds_tensor = torch.stack([torch.tensor(x) for x in class_preds])
            clabels_tensor = torch.stack([torch.tensor(x) for x in clabels])
            y_probs_tensor = torch.stack([torch.tensor(x) for x in y_probs])

            acc_class = acc(class_preds_tensor, clabels_tensor).item()
            prec_class = prec(class_preds_tensor, clabels_tensor).item()
            rec_class = rec(class_preds_tensor, clabels_tensor).item()
            f1_class = f1(class_preds_tensor, clabels_tensor).item()
            auroc_class = auroc(y_probs_tensor, clabels_tensor).item()

            # Bootstrap 置信区间
            class_preds_np = np.concatenate([np.atleast_1d(x) for x in class_preds])
            clabels_np = np.concatenate([np.atleast_1d(x) for x in clabels])
            y_probs_np = np.concatenate([np.atleast_1d(x) for x in y_probs])

            metrics_ci = compute_cls_ci_bootstrap(clabels_np, class_preds_np, y_prob=y_probs_np, n_bootstraps=1000)

            acc_ci = metrics_ci['accuracy'][1]
            prec_ci = metrics_ci['precision'][1]
            rec_ci = metrics_ci['recall'][1]
            f1_ci = metrics_ci['f1'][1]
            auroc_ci = metrics_ci['auroc'][1]
            spec_class, spec_ci = metrics_ci['specificity']

            # 打印 & 写日志
            print("Class test of best model:")
            print(f"  acc_class: {acc_class:.4f} ± {acc_ci:.4f}")
            print(f"  prec_class: {prec_class:.4f} ± {prec_ci:.4f}")
            print(f"  rec_class: {rec_class:.4f} ± {rec_ci:.4f}")
            print(f"  specificity_class: {spec_class:.4f} ± {spec_ci:.4f}")
            print(f"  f1_class: {f1_class:.4f} ± {f1_ci:.4f}")
            print(f"  auroc_class: {auroc_class:.4f} ± {auroc_ci:.4f}")

            log_info = ""
            log_info += "Class test of best model:\n"
            log_info += f"  acc_class: {acc_class:.4f} ± {acc_ci:.4f}\n"
            log_info += f"  prec_class: {prec_class:.4f} ± {prec_ci:.4f}\n"
            log_info += f"  rec_class: {rec_class:.4f} ± {rec_ci:.4f}\n"
            log_info += f"  specificity_class: {spec_class:.4f} ± {spec_ci:.4f}\n"
            log_info += f"  f1_class: {f1_class:.4f} ± {f1_ci:.4f}\n"
            log_info += f"  auroc_class: {auroc_class:.4f} ± {auroc_ci:.4f}\n"

            with open(os.path.join(result_path, 'log', 'metrics_log.txt'), 'a') as f:
                f.write(log_info + '\n')

        else:
            auroc_class = 1e-8

    return np.mean(loss_list)


if __name__ == '__main__':
    
    masks_path = "results/no_ccaff_NoK_caro_no_ccaff_seg/background/"
    result_path = "results/no_ccaff_NoK_caro_no_ccaff_seg/"
    model_path = 'results/no_ccaff_NoK_caro_no_ccaff_seg/checkpoints/best-epoch233-loss0.8424.pth'

    directory = os.path.dirname(result_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

    config = setting_config
    device_ids = config.device_ids
    criterion = config.criterion
    criterion_class = config.criterion_class

    log_dir = os.path.join(result_path, 'log')

    device = torch.device('cuda:{}'.format(device_ids[0]) if torch.cuda.is_available() else 'cpu')
    if config.network == 'CMENet':
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

    elif config.network == 'UNet':
        from models.compare_methods.UNet_for_Car import UNet
        network = UNet(config=config, in_ch=30, out_ch=30)
        
    elif config.network == 'AttUNet':
        from models.compare_methods.Attunet_for_Car import AttU_Net
        network = AttU_Net(config=config,in_channel=30, out_channel=30)
        
    elif config.network == 'CASF':
        from models.compare_methods.CASF.lib.TransFuse import TransFuse_S as CASF
        network = CASF(config=config)
    
    elif config.network == 'UNetv2':
        from models.compare_methods.unet_v2.UNet_v2 import UNetV2
        network = UNetV2(config=config, in_channels=30, out_channels=30)
        
    elif config.network == 'NestedUNet':
        from models.compare_methods.NestedUNet import NestedUNet
        network = NestedUNet(config=config)
        
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
    # elif config.network == 'CASF':
    #     network = CASF()
    # elif config.network == 'FAT_Net':
    #     network = FAT_Net()
    model = nn.DataParallel(network, device_ids=device_ids)
    model.to(device)


    best_weight = torch.load(model_path, map_location=device)
    model.load_state_dict(best_weight)

    test_dataset, dataset = load_data(config, config.xlsx_dir, config.test_txt_file, config.txt_file, config.images_dir, config.masks_dir, config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)


    loss = test_one_epoch(
                        test_loader,
                        model,
                        criterion,
                        criterion_class,
                        config,
                        device,
                        result_path,
                    )
