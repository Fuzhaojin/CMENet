# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project

import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import *
import time
import pdb
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import torch.nn.functional as F

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

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    criterion_class,
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config,
                    device,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        # images, targets = data
        t1, t1c, t2, tof, targets, clabel, seg_result, _ = data
        # images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        t1 = t1.to(device, dtype=torch.float)
        t1c = t1c.to(device, dtype=torch.float)
        # t2 = t2.to(device, dtype=torch.float)
        tof = tof.to(device, dtype=torch.float)
        seg_result = seg_result.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        clabel = clabel.to(device, dtype=torch.long)

        if config.amp:
            with autocast():
                out, cls_out = model(t1, t1c, tof, config.stage)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # out, t1_class, t1c_class, t2_class, tof_class, cls_out = model(t1, t1c, t2, tof, seg_result, config.stage)
            if config.network == "AttUNet" or config.network == "CASF" or config.network == "UNetv2" or config.network == 'NestedUNet':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, config.stage)

            elif config.network == "nnUnet":
                out = model(t1, t1c, tof, config.stage)
            
            elif config.network == "UNet" or config.network == "CMENet" or config.network == 'final_use_seg':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, seg_result, config.stage)
            
            elif config.network == 'ResNet50' or config.network == 'ResNet101' or config.network == 'Densenet121' or config.network == 'Vit' or config.network == 'Efficientnet_b3':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, seg_result, config.stage)
                
                
            if config.network == 'xr_t1':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, config.stage)
            elif config.network == 'xr_t1c':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1c, config.stage)
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
            
            # t1_class = torch.sigmoid(t1_class)
            # t1c_class = torch.sigmoid(t1c_class)
            # t2_class = torch.sigmoid(t2_class)
            # tof_class = torch.sigmoid(tof_class)
            # y_prob = torch.sigmoid(cls_out)  # 转换为概率
            # cls_out = y_prob
            
            if config.stage == 'seg':
                # 训练分割模型
                loss_mask = criterion(out, targets)
                loss = loss_mask
            elif config.stage == 'cls':
                # 训练分类模型
                # import pdb; pdb.set_trace()
                loss_t1 = criterion_class(t1_class, clabel)
                loss_t1c = criterion_class(t1c_class, clabel)
                # loss_t2 = criterion_class(t2_class, clabel) 
                loss_tof = criterion_class(tof_class, clabel)  
                loss_class = criterion_class(cls_out, clabel)
                loss = loss_class + loss_t1 + loss_t1c + loss_tof
            else:
                # 默认分割模型
                loss_mask = criterion(out, targets)
                loss = loss_mask 
            
            loss.backward()
            optimizer.step()
            
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 


def val_one_epoch(test_loader,
                    model,
                    criterion,
                    criterion_class, 
                    epoch, 
                    logger,
                    config,
                    device
                  ):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    # 分类计数器
    class_preds = []
    clabels = []
    y_probs = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            t1, t1c, t2, tof, targets, clabel, seg_result, _ = data
            t1 = t1.to(device, dtype=torch.float)
            t1c = t1c.to(device, dtype=torch.float)
            # t2 = t2.to(device, dtype=torch.float)
            tof = tof.to(device, dtype=torch.float)
            seg_result = seg_result.to(device, dtype=torch.float)
            
            targets = targets.to(device, dtype=torch.float)
            clabel = clabel.to(device, dtype=torch.long)

            
            # 模型选择
            if config.network == "AttUNet" or config.network == "CASF" or config.network == "UNetv2" or config.network == 'NestedUNet':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, config.stage)

            elif config.network == "nnUnet":
                out = model(t1, t1c, tof, config.stage)
            
            elif config.network == "UNet" or config.network == "CMENet" or config.network == 'final_use_seg':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, seg_result, config.stage)
            
            elif config.network == 'ResNet50' or config.network == 'ResNet101' or config.network == 'Densenet121' or config.network == 'Vit' or config.network == 'Efficientnet_b3':
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
                # print(t1_class.shape)
                # print(clabel.shape)
                loss_t1 = criterion_class(t1_class, clabel)
                loss_t1c = criterion_class(t1c_class, clabel)
                # loss_t2 = criterion_class(t2_class, clabel) 
                loss_tof = criterion_class(tof_class, clabel)  
                loss_class = criterion_class(cls_out, clabel)
                loss = loss_class + loss_t1 + loss_t1c + loss_tof  

                # 获取概率和预测类别
                y_prob = torch.softmax(cls_out, dim=1)[:, 1]  # 二分类，取类别 1 的概率
                predict_y = (y_prob > 0.5).long()  # 阈值划分为 0 或 1
                
                # 汇总分类结果
                class_preds.append(predict_y)
                clabels.append(clabel)
                y_probs.append(y_prob)
                
            else:
                # 默认分割模型
                loss_mask = criterion(out, targets)
                loss = loss_mask 
            
            # 汇总分割结果
            gts.append(targets.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            
            loss_list.append(loss.item())
            

    if epoch % config.val_interval == 0:
        if config.stage == 'seg':
            preds = [np.array(p).reshape(-1) for p in preds if isinstance(p, (list, np.ndarray))]
            preds = np.concatenate(preds)

            gts = [np.array(p).reshape(-1) for p in gts if isinstance(p, (list, np.ndarray))]
            gts = np.concatenate(gts)

            # preds = np.array(preds).reshape(-1)
            # gts = np.array(gts).reshape(-1)

            y_pre = np.where(preds>=0.5, 1, 0)
            y_true = np.where(gts>=0.5, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            specificity_mask = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                    specificity: {specificity_mask}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
            print(log_info)
            logger.info(log_info)
        
        if config.stage == 'cls':
            # 转换为 tensor 并保持在 GPU 上
            class_preds_tensor = torch.cat(class_preds).to(device)  # [N_total]
            clabels_tensor = torch.cat(clabels).to(device)          # [N_total]
            y_probs_tensor = torch.cat(y_probs).to(device)          # [N_total]
            
            # 分类指标：
            # 初始化指标计算器
            acc = Accuracy(task="binary").to(device)
            prec = Precision(task="binary").to(device)
            rec = Recall(task="binary").to(device)      # 即Sensitivity
            f1 = F1Score(task="binary").to(device)
            auroc = AUROC(task="binary").to(device)
            
            acc_class = acc(class_preds_tensor, clabels_tensor)
            prec_class = prec(class_preds_tensor, clabels_tensor)
            rec_class = rec(class_preds_tensor, clabels_tensor)
            f1_class = f1(class_preds_tensor, clabels_tensor)
            auroc_class = auroc(y_probs_tensor, clabels_tensor)

            # class
            log_info = f'acc_class: {acc_class}, prec_class: {prec_class}, rec_class: {rec_class}, f1_class: {f1_class}, auroc_class: {auroc_class}'
            print(log_info)
            logger.info(log_info)
        else:
            auroc_class = 1e-8
            acc_class = 1e-8

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
        
    if config.stage == 'cls':
        return np.mean(loss_list), auroc_class, acc_class
    elif config.stage == 'seg':
        return np.mean(loss_list), f1_or_dsc


import time
def test_one_epoch(test_loader,
                    model,
                    criterion,
                    criterion_class,
                    logger,
                    config,
                    device,
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

            elif config.network == "nnUnet":
                out = model(t1, t1c, tof, config.stage)
            
            elif config.network == "UNet" or config.network == "CMENet" or config.network == 'final_use_seg':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, seg_result, config.stage)
            
            elif config.network == 'ResNet50' or config.network == 'ResNet101' or config.network == 'Densenet121' or config.network == 'Vit' or config.network == 'Efficientnet_b3':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, t1c, tof, seg_result, config.stage)


            if config.network == 'xr_t1':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1, config.stage)
            elif config.network == 'xr_t1c':
                out, t1_class, t1c_class, tof_class, cls_out = model(t1c, config.stage)
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
                
            else:
                # 默认分割模型
                loss_mask = criterion(out, targets)
                loss = loss_mask 
            
            
            # 分割结果汇总
            targets = targets.squeeze(1).cpu().detach().numpy()
            gts.append(targets)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)     
               
            loss_list.append(loss.item())
            

            # print(out.shape)
            # pdb.set_trace()
            if i % config.save_interval == 0:
                # save_imgs(img, targets, out, i, config.work_dir + 'original/', config.threshold, test_data_name=test_data_name)
                save_nii_msk(targets, out, i, name, config.work_dir + 'background/', config.threshold,test_data_name=test_data_name)
                if config.stage == 'cls':
                    save_nii_mskpred(out, i, name, config.work_dir + 'outputs/', config.threshold,test_data_name=test_data_name)
                if config.stage == 'cls':
                    save_class_out(predict_y, clabel, i, name, config.file_name, config.work_dir + 'class_outputs/', config.threshold, test_data_name=test_data_name, y_prob=y_prob)
        
        if config.stage == 'seg':  
            preds = np.array(preds).reshape(-1)
            gts = np.array(gts).reshape(-1)

            y_pre = np.where(preds>=0.5, 1, 0)
            y_true = np.where(gts>=0.5, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            f1_or_dsc = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            miou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

            # 存储与打印指标
            if test_data_name is not None:
                log_info = f'test_datasets_name: {test_data_name}'
                print(log_info)
                logger.info(log_info)
                
            log_info = f'Mask test of best model, loss: {np.mean(loss_list):.4f}, ' \
                        f'miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, ' \
                        f'precision: {precision:.4f}, recall: {recall:.4f}, confusion_matrix: {confusion.tolist()}\n'
            print(log_info)
            logger.info(log_info)

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
            logger.info(log_info)
            
            # 追加写入 metrics_log.txt 文件
            with open(os.path.join(config.work_dir, 'log', 'metrics_log.txt'), 'a') as f:
                f.write(log_info + '\n')
            
        else:
            auroc_class = 1e-8

    return np.mean(loss_list)
