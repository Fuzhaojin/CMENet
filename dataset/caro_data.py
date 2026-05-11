# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project

import os
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import nibabel as nib
from torch.utils.data import DataLoader
import pandas as pd
import pdb

class CustomTransform:
    def __init__(self, normalize=True, flip=False, rotate=False):
        self.normalize = normalize
        self.flip = flip
        self.rotate = rotate

    def __call__(self, t1, t1c, t2, tof, mask):
        # 归一化
        if self.normalize:
            t1 = self.normalize_image(t1)
            t1c = self.normalize_image(t1c)
            t2 = self.normalize_image(t2)
            tof = self.normalize_image(tof)
            mask = self.normalize_image(mask)

        # 翻转
        if self.flip and np.random.rand() > 0.5:
            t1 = F.hflip(t1)
            t1c = F.hflip(t1c)
            t2 = F.hflip(t2)
            tof = F.hflip(tof)
            mask = F.hflip(mask)

        # 旋转
        if self.rotate:
            angle = np.random.choice([0.0, 90.0, 180.0, 270.0])
            t1 = F.rotate(t1, angle)
            t1c = F.rotate(t1c, angle)
            t2 = F.rotate(t2, angle)
            tof = F.rotate(tof, angle)
            mask = F.rotate(mask, angle)

        return t1, t1c, t2, tof, mask

    def normalize_image(self, image):
        # 归一化到 [0, 1] 范围
        image = (image - image.min()) / (image.max() - image.min())
        return image

class CARO_datasets(Dataset):
    def __init__(self, txt_file, images_dir, masks_dir, transform=None, train=True):
        """
        Args:
            txt_file (string): 包含文件名的txt文件路径。
            images_dir (string): 包含图像数据的文件夹路径。
            masks_dir (string): 包含掩码数据的文件夹路径。
            transform (callable, optional): 可选的变换函数，用于对样本进行预处理。
        """
        self.txt_file = txt_file
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.train = train
        
        # 读取txt文件中的文件名
        with open(self.txt_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        self.T1_path = os.path.join(self.images_dir, "T1")
        self.T1C_path = os.path.join(self.images_dir, "T1C")
        self.T2_path = os.path.join(self.images_dir, "T2")
        self.TOF_path = os.path.join(self.images_dir, "TOF")

        self.data = []

        for i in range(len(self.file_names)):
            t1_path = self.T1_path + "/" + self.file_names[i] + "_image.nii"
            t1c_path = self.T1C_path + "/" + self.file_names[i] + "_image.nii"
            t2_path = self.T2_path + "/" + self.file_names[i] + "_image.nii"
            tof_path = self.TOF_path + "/" + self.file_names[i] + "_image.nii"
            mask_path = self.masks_dir + "/" + self.file_names[i] + "_label.nii"
            name = self.file_names[i]
            self.data.append({"t1": t1_path,
                              "t1c": t1c_path,
                              "t2": t2_path,
                              "tof": tof_path,
                              "mask": mask_path,
                              "name": name
                              })

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        # 使用nibabel加载nii格式的数据
        t1_nii = nib.load(data_dict["t1"])
        t1c_nii = nib.load(data_dict["t1c"])
        t2_nii = nib.load(data_dict["t2"])
        tof_nii = nib.load(data_dict["tof"])
        mask_nii = nib.load(data_dict["mask"])
        
        t1 = t1_nii.get_fdata()
        t1c = t1c_nii.get_fdata()
        t2 = t2_nii.get_fdata()
        tof = tof_nii.get_fdata()
        mask = mask_nii.get_fdata()
        
        # # 将数据转换为PyTorch张量
        # t1 = torch.tensor(t1, dtype=torch.float32)
        # t1c = torch.tensor(t1c, dtype=torch.float32)
        # t2 = torch.tensor(t2, dtype=torch.float32)
        # tof = torch.tensor(tof, dtype=torch.float32)
        # mask = torch.tensor(mask, dtype=torch.float32)

        # t1 = t1.permute(2, 0, 1)
        # t1c = t1c.permute(2, 0, 1)
        # t2 = t2.permute(2, 0, 1)
        # tof = tof.permute(2, 0, 1)
        # mask = mask.permute(2, 0, 1)

        # 可选的变换
        if self.transform and self.train:
            t1, t1c, t2, tof, mask = self.transform((t1, t1c, t2, tof, mask))
        
        return t1, t1c, t2, tof, mask, data_dict["name"]
    

class CARO_datasets_Xlsx(Dataset):
    def __init__(self, xlsx_dir, txt_file, images_dir, masks_dir, transform=None, train=True):
        """
        Args:
            txt_file (string): 包含文件名的txt文件路径。
            images_dir (string): 包含图像数据的文件夹路径。
            masks_dir (string): 包含掩码数据的文件夹路径。
            transform (callable, optional): 可选的变换函数，用于对样本进行预处理。
        """
        self.xlsx_dir = xlsx_dir
        self.txt_file = txt_file
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.train = train
        
        # 读取 表格 文件中的文件名
        df = pd.read_excel(self.xlsx_dir, engine="openpyxl")
        filenames = df["name"].tolist()
        class_labels = df['class_label'].astype(int).tolist()  # 第二列转为整型列表
        file_label_dict = dict(zip(filenames, class_labels))
        
        with open(self.txt_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        self.T1_path = os.path.join(self.images_dir, "T1")
        self.T1C_path = os.path.join(self.images_dir, "T1C")
        self.T2_path = os.path.join(self.images_dir, "T2")
        self.TOF_path = os.path.join(self.images_dir, "TOF")
        self.outputs_masks_path = os.path.join(self.images_dir, "seg_result")  # 自己模型使用的
        

        self.data = []
    

        for i in range(len(self.file_names)):
            # CarotidData
            t1_path = self.T1_path + "/" + self.file_names[i] + "_image.nii"
            t1c_path = self.T1C_path + "/" + self.file_names[i] + "_image.nii"
            t2_path = self.T2_path + "/" + self.file_names[i] + "_image.nii"
            tof_path = self.TOF_path + "/" + self.file_names[i] + "_image.nii"
            
            mask_path = self.masks_dir + "/" + self.file_names[i] + "_label.nii"
            seg_result_path = self.outputs_masks_path + "/" + self.file_names[i] + "_label.nii"  # _pred.nii
            class_label = file_label_dict[self.file_names[i]] - 1.0
            
            # class_label = file_label_dict[self.file_names[i]]
            # t2_path = t1_path
            # seg_result_path = mask_path

            # new caro or final
            # t1_path = self.T1_path + "/" + self.file_names[i] + "T1.nii"
            # t1c_path = self.T1C_path + "/" + self.file_names[i] + "T1C.nii"
            # t2_path = self.T2_path + "/" + self.file_names[i] + "T2.nii"
            # tof_path = self.TOF_path + "/" + self.file_names[i] + "TOF.nii"
            # mask_path = self.masks_dir + "/T1/" + self.file_names[i] + "T1.nii"
            # class_label = file_label_dict[self.file_names[i]] - 1.0


            name = self.file_names[i]
            self.data.append({"t1": t1_path,
                              "t1c": t1c_path,
                              "t2": t2_path,
                              "tof": tof_path,
                              "seg_result": seg_result_path,
                              "mask": mask_path,
                              "class":class_label,
                              "name": name
                              })

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        # 使用nibabel加载nii格式的数据
        t1_nii = nib.load(data_dict["t1"])
        t1c_nii = nib.load(data_dict["t1c"])
        t2_nii = nib.load(data_dict["t2"])
        tof_nii = nib.load(data_dict["tof"])
        seg_result_nii = nib.load(data_dict["seg_result"])
        mask_nii = nib.load(data_dict["mask"])
        
        t1 = t1_nii.get_fdata()
        t1c = t1c_nii.get_fdata()
        t2 = t2_nii.get_fdata()
        tof = tof_nii.get_fdata()
        seg_result = seg_result_nii.get_fdata()
        mask = mask_nii.get_fdata()
        

        # 可选的变换
        if self.transform and self.train:
            t1, t1c, t2, tof, mask, seg_result = self.transform((t1, t1c, t2, tof, mask, seg_result))
            # print(mask.shape)
            # print(t1.shape)
        return t1, t1c, t2, tof, mask, data_dict['class'], seg_result, data_dict["name"]



class crop30_dataset(Dataset):
    def __init__(self, xlsx_dir, txt_file, images_dir, masks_dir, transform=None, train=True):
        """
        Args:
            txt_file (string): 包含文件名的txt文件路径。
            images_dir (string): 包含图像数据的文件夹路径。
            masks_dir (string): 包含掩码数据的文件夹路径。
            transform (callable, optional): 可选的变换函数，用于对样本进行预处理。
        """
        self.xlsx_dir = xlsx_dir
        self.txt_file = txt_file
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.train = train
        
        # 读取 表格 文件中的文件名
        df = pd.read_excel(self.xlsx_dir, engine="openpyxl")
        filenames = df["name"].tolist()
        class_labels = df['class_label'].astype(int).tolist()  # 第二列转为整型列表
        file_label_dict = dict(zip(filenames, class_labels))
        
        with open(self.txt_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        self.T1_path = os.path.join(self.images_dir, "T1")
        self.T2_path = os.path.join(self.images_dir, "T1")
        self.T1C_path = os.path.join(self.images_dir, "T1C")
        self.TOF_path = os.path.join(self.images_dir, "TOF")
        # self.outputs_masks_path = os.path.join(self.masks_dir, "outputs_masks_T1")

        self.data = []
    

        for i in range(len(self.file_names)):
            # CarotidData
            t1_path = self.T1_path + "/" + self.file_names[i] + "_T1.nii"
            t1c_path = self.T1C_path + "/" + self.file_names[i] + "_T1C.nii"
            tof_path = self.TOF_path + "/" + self.file_names[i] + "_TOF.nii"
            # seg_result_path = self.outputs_masks_path + "/" + self.file_names[i] + "_T1.nii"  # _pred.nii
            # mask_path = self.masks_dir + "/" + self.file_names[i] + "_T1.nii"
            
            # seg_result_path = t1_path
            # mask_path = t1_path
            seg_result_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"  # _pred.nii
            mask_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"
            
            t2_path = self.T1_path + "/" + self.file_names[i] + "_T1.nii"
            
            class_label = file_label_dict[self.file_names[i]]
            
            
            
            # class_label = file_label_dict[self.file_names[i]]

            # new caro or final
            # t1_path = self.T1_path + "/" + self.file_names[i] + "T1.nii"
            # t1c_path = self.T1C_path + "/" + self.file_names[i] + "T1C.nii"
            # t2_path = self.T2_path + "/" + self.file_names[i] + "T2.nii"
            # tof_path = self.TOF_path + "/" + self.file_names[i] + "TOF.nii"
            # mask_path = self.masks_dir + "/T1/" + self.file_names[i] + "T1.nii"
            # class_label = file_label_dict[self.file_names[i]] - 1.0


            name = self.file_names[i]
            self.data.append({"t1": t1_path,
                              "t1c": t1c_path,
                              "tof": tof_path,
                              "seg_result": seg_result_path,
                              "mask": mask_path,
                              "class":class_label,
                              "name": name,
                              "t2": t2_path
                              })

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        # 使用nibabel加载nii格式的数据
        t1_nii = nib.load(data_dict["t1"])
        t2_nii = nib.load(data_dict["t2"])
        t1c_nii = nib.load(data_dict["t1c"])
        tof_nii = nib.load(data_dict["tof"])
        seg_result_nii = nib.load(data_dict["seg_result"])
        mask_nii = nib.load(data_dict["mask"])
        
        t1 = t1_nii.get_fdata()
        t2 = t2_nii.get_fdata()
        t1c = t1c_nii.get_fdata()
        tof = tof_nii.get_fdata()
        seg_result = seg_result_nii.get_fdata()
        mask = mask_nii.get_fdata()
        
        # # 将数据转换为PyTorch张量
        # t1 = torch.tensor(t1, dtype=torch.float32)
        # t1c = torch.tensor(t1c, dtype=torch.float32)
        # tof = torch.tensor(tof, dtype=torch.float32)
        # mask = torch.tensor(mask, dtype=torch.float32)

        # t1 = t1.permute(2, 0, 1)
        # t1c = t1c.permute(2, 0, 1)
        # tof = tof.permute(2, 0, 1)
        # mask = mask.permute(2, 0, 1)

        # 可选的变换
        if self.transform and self.train:
            t1, t1c, t2, tof, mask, seg_result = self.transform((t1, t1c, t2, tof, mask, seg_result))
            # print(mask.shape)
            # print(t1.shape)

        return t1, t1c, t2, tof, mask, data_dict['class'], seg_result, data_dict["name"]


class weifang_dataset(Dataset):
    def __init__(self, xlsx_dir, txt_file, images_dir, masks_dir, transform=None, train=True):
        """
        Args:
            txt_file (string): 包含文件名的txt文件路径。
            images_dir (string): 包含图像数据的文件夹路径。
            masks_dir (string): 包含掩码数据的文件夹路径。
            transform (callable, optional): 可选的变换函数，用于对样本进行预处理。
        """
        self.xlsx_dir = xlsx_dir
        self.txt_file = txt_file
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.train = train
        
        # 读取 表格 文件中的文件名
        df = pd.read_excel(self.xlsx_dir, engine="openpyxl")
        filenames = df["name"].tolist()
        class_labels = df['class_label'].astype(int).tolist()  # 第二列转为整型列表
        file_label_dict = dict(zip(filenames, class_labels))
        
        with open(self.txt_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        self.T1_path = os.path.join(self.images_dir, "T1")
        self.T2_path = os.path.join(self.images_dir, "T1")
        self.T1C_path = os.path.join(self.images_dir, "T1C")
        self.TOF_path = os.path.join(self.images_dir, "TOF")
        # self.outputs_masks_path = os.path.join(self.images_dir, "outputs_masks_T1")

        self.data = []
    

        for i in range(len(self.file_names)):
            # CarotidData
            t1_path = self.T1_path + "/" + self.file_names[i] + "_T1.nii"
            t2_path = self.T1_path + "/" + self.file_names[i] + "_T1.nii"
            t1c_path = self.T1C_path + "/" + self.file_names[i] + "_T1C.nii"
            tof_path = self.TOF_path + "/" + self.file_names[i] + "_TOF.nii"
            # seg_result_path = self.outputs_masks_path + "/" + self.file_names[i] + "_T1.nii"  # _pred.nii
            # mask_path = self.masks_dir + "/" + self.file_names[i] + "_T1.nii"
            
            seg_result_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"  # _pred.nii
            mask_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"
            
            class_label = file_label_dict[self.file_names[i]]
            
            
            
            # class_label = file_label_dict[self.file_names[i]]

            # new caro or final
            # t1_path = self.T1_path + "/" + self.file_names[i] + "T1.nii"
            # t1c_path = self.T1C_path + "/" + self.file_names[i] + "T1C.nii"
            # t2_path = self.T2_path + "/" + self.file_names[i] + "T2.nii"
            # tof_path = self.TOF_path + "/" + self.file_names[i] + "TOF.nii"
            # mask_path = self.masks_dir + "/T1/" + self.file_names[i] + "T1.nii"
            # class_label = file_label_dict[self.file_names[i]] - 1.0


            name = self.file_names[i]
            self.data.append({"t1": t1_path,
                              "t2": t2_path,
                              "t1c": t1c_path,
                              "tof": tof_path,
                              "seg_result": seg_result_path,
                              "mask": mask_path,
                              "class":class_label,
                              "name": name
                              })

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        # 使用nibabel加载nii格式的数据
        t1_nii = nib.load(data_dict["t1"])
        t2_nii = nib.load(data_dict["t2"])
        t1c_nii = nib.load(data_dict["t1c"])
        tof_nii = nib.load(data_dict["tof"])
        seg_result_nii = nib.load(data_dict["seg_result"])
        mask_nii = nib.load(data_dict["mask"])
        
        t1 = t1_nii.get_fdata()
        t2 = t2_nii.get_fdata()
        t1c = t1c_nii.get_fdata()
        tof = tof_nii.get_fdata()
        seg_result = seg_result_nii.get_fdata()
        mask = mask_nii.get_fdata()
        
        # # 将数据转换为PyTorch张量
        # t1 = torch.tensor(t1, dtype=torch.float32)
        # t1c = torch.tensor(t1c, dtype=torch.float32)
        # t2 = torch.tensor(t2, dtype=torch.float32)
        # tof = torch.tensor(tof, dtype=torch.float32)
        # mask = torch.tensor(mask, dtype=torch.float32)

        # t1 = t1.permute(2, 0, 1)
        # t1c = t1c.permute(2, 0, 1)
        # t2 = t2.permute(2, 0, 1)
        # tof = tof.permute(2, 0, 1)
        # mask = mask.permute(2, 0, 1)

        # 可选的变换
        if self.transform and self.train:
            t1, t1c, t2, tof, mask, seg_result = self.transform((t1, t1c, t2, tof, mask, seg_result))
            # print(mask.shape)
            # print(t1.shape)

        return t1, t1c, t2, tof, mask, data_dict['class'], seg_result, data_dict["name"]


class QZ_dataset(Dataset):
    def __init__(self, xlsx_dir, txt_file, images_dir, masks_dir, transform=None, train=True):
        """
        Args:
            txt_file (string): 包含文件名的txt文件路径。
            images_dir (string): 包含图像数据的文件夹路径。
            masks_dir (string): 包含掩码数据的文件夹路径。
            transform (callable, optional): 可选的变换函数，用于对样本进行预处理。
        """
        self.xlsx_dir = xlsx_dir
        self.txt_file = txt_file
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.train = train
        
        # 读取 表格 文件中的文件名
        df = pd.read_excel(self.xlsx_dir, engine="openpyxl")
        filenames = df["name"].tolist()
        class_labels = df['class_label'].astype(int).tolist()  # 第二列转为整型列表
        file_label_dict = dict(zip(filenames, class_labels))
        
        with open(self.txt_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        self.T1_path = os.path.join(self.images_dir, "T1")
        self.T2_path = os.path.join(self.images_dir, "T1")
        self.T1C_path = os.path.join(self.images_dir, "T1C")
        self.TOF_path = os.path.join(self.images_dir, "TOF")
        # self.outputs_masks_path = os.path.join(self.images_dir, "outputs_masks_T1")

        self.data = []
    

        for i in range(len(self.file_names)):
            # CarotidData
            t1_path = self.T1_path + "/" + self.file_names[i] + "_T1.nii"
            t2_path = self.T1_path + "/" + self.file_names[i] + "_T1.nii"
            t1c_path = self.T1C_path + "/" + self.file_names[i] + "_T1C.nii"
            tof_path = self.TOF_path + "/" + self.file_names[i] + "_TOF.nii"
            # seg_result_path = self.outputs_masks_path + "/" + self.file_names[i] + "_T1.nii"  # _pred.nii
            mask_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"
            
            seg_result_path = mask_path
            # seg_result_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"  # _pred.nii
            # mask_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"
            
            class_label = file_label_dict[self.file_names[i]]
            
            
            
            # class_label = file_label_dict[self.file_names[i]]

            # new caro or final
            # t1_path = self.T1_path + "/" + self.file_names[i] + "T1.nii"
            # t1c_path = self.T1C_path + "/" + self.file_names[i] + "T1C.nii"
            # t2_path = self.T2_path + "/" + self.file_names[i] + "T2.nii"
            # tof_path = self.TOF_path + "/" + self.file_names[i] + "TOF.nii"
            # mask_path = self.masks_dir + "/T1/" + self.file_names[i] + "T1.nii"
            # class_label = file_label_dict[self.file_names[i]] - 1.0


            name = self.file_names[i]
            self.data.append({"t1": t1_path,
                              "t2": t2_path,
                              "t1c": t1c_path,
                              "tof": tof_path,
                              "seg_result": seg_result_path,
                              "mask": mask_path,
                              "class":class_label,
                              "name": name
                              })

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        # 使用nibabel加载nii格式的数据
        t1_nii = nib.load(data_dict["t1"])
        t2_nii = nib.load(data_dict["t2"])
        t1c_nii = nib.load(data_dict["t1c"])
        tof_nii = nib.load(data_dict["tof"])
        seg_result_nii = nib.load(data_dict["seg_result"])
        mask_nii = nib.load(data_dict["mask"])
        
        t1 = t1_nii.get_fdata()
        t2 = t2_nii.get_fdata()
        t1c = t1c_nii.get_fdata()
        tof = tof_nii.get_fdata()
        seg_result = seg_result_nii.get_fdata()
        mask = mask_nii.get_fdata()
        
        # # 将数据转换为PyTorch张量
        # t1 = torch.tensor(t1, dtype=torch.float32)
        # t1c = torch.tensor(t1c, dtype=torch.float32)
        # t2 = torch.tensor(t2, dtype=torch.float32)
        # tof = torch.tensor(tof, dtype=torch.float32)
        # mask = torch.tensor(mask, dtype=torch.float32)

        # t1 = t1.permute(2, 0, 1)
        # t1c = t1c.permute(2, 0, 1)
        # t2 = t2.permute(2, 0, 1)
        # tof = tof.permute(2, 0, 1)
        # mask = mask.permute(2, 0, 1)

        # 可选的变换
        if self.transform and self.train:
            t1, t1c, t2, tof, mask, seg_result = self.transform((t1, t1c, t2, tof, mask, seg_result))
            # print(mask.shape)
            # print(t1.shape)

        return t1, t1c, t2, tof, mask, data_dict['class'], seg_result, data_dict["name"]

class SZ_dataset(Dataset):
    def __init__(self, xlsx_dir, txt_file, images_dir, masks_dir, transform=None, train=True):
        """
        Args:
            txt_file (string): 包含文件名的txt文件路径。
            images_dir (string): 包含图像数据的文件夹路径。
            masks_dir (string): 包含掩码数据的文件夹路径。
            transform (callable, optional): 可选的变换函数，用于对样本进行预处理。
        """
        self.xlsx_dir = xlsx_dir
        self.txt_file = txt_file
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.train = train
        
        # 读取 表格 文件中的文件名
        df = pd.read_excel(self.xlsx_dir, engine="openpyxl")
        filenames = df["name"].tolist()
        class_labels = df['class_label'].astype(int).tolist()  # 第二列转为整型列表
        file_label_dict = dict(zip(filenames, class_labels))
        
        with open(self.txt_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        self.T1_path = os.path.join(self.images_dir, "T1")
        self.T2_path = os.path.join(self.images_dir, "T1")
        self.T1C_path = os.path.join(self.images_dir, "T1C")
        self.TOF_path = os.path.join(self.images_dir, "TOF")
        # self.outputs_masks_path = os.path.join(self.images_dir, "outputs_masks_T1")

        self.data = []
    

        for i in range(len(self.file_names)):
            # CarotidData
            t1_path = self.T1_path + "/" + self.file_names[i] + "_T1.nii"
            t2_path = self.T1_path + "/" + self.file_names[i] + "_T1.nii"
            t1c_path = self.T1C_path + "/" + self.file_names[i] + "_T1C.nii"
            tof_path = self.TOF_path + "/" + self.file_names[i] + "_TOF.nii"
            
            # mask_path = t1_path
            # seg_result_path = mask_path 
            
            seg_result_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"  # _pred.nii
            mask_path = self.masks_dir + "/" + self.file_names[i] + "_pred.nii"
            
            class_label = file_label_dict[self.file_names[i]]
            
            
            
            # class_label = file_label_dict[self.file_names[i]]

            # new caro or final
            # t1_path = self.T1_path + "/" + self.file_names[i] + "T1.nii"
            # t1c_path = self.T1C_path + "/" + self.file_names[i] + "T1C.nii"
            # t2_path = self.T2_path + "/" + self.file_names[i] + "T2.nii"
            # tof_path = self.TOF_path + "/" + self.file_names[i] + "TOF.nii"
            # mask_path = self.masks_dir + "/T1/" + self.file_names[i] + "T1.nii"
            # class_label = file_label_dict[self.file_names[i]] - 1.0


            name = self.file_names[i]
            self.data.append({"t1": t1_path,
                              "t2": t2_path,
                              "t1c": t1c_path,
                              "tof": tof_path,
                              "seg_result": seg_result_path,
                              "mask": mask_path,
                              "class":class_label,
                              "name": name
                              })

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        # 使用nibabel加载nii格式的数据
        t1_nii = nib.load(data_dict["t1"])
        t2_nii = nib.load(data_dict["t2"])
        t1c_nii = nib.load(data_dict["t1c"])
        tof_nii = nib.load(data_dict["tof"])
        seg_result_nii = nib.load(data_dict["seg_result"])
        mask_nii = nib.load(data_dict["mask"])
        
        t1 = t1_nii.get_fdata()
        t2 = t2_nii.get_fdata()
        t1c = t1c_nii.get_fdata()
        tof = tof_nii.get_fdata()
        seg_result = seg_result_nii.get_fdata()
        mask = mask_nii.get_fdata()
        
        # # 将数据转换为PyTorch张量
        # t1 = torch.tensor(t1, dtype=torch.float32)
        # t1c = torch.tensor(t1c, dtype=torch.float32)
        # t2 = torch.tensor(t2, dtype=torch.float32)
        # tof = torch.tensor(tof, dtype=torch.float32)
        # mask = torch.tensor(mask, dtype=torch.float32)

        # t1 = t1.permute(2, 0, 1)
        # t1c = t1c.permute(2, 0, 1)
        # t2 = t2.permute(2, 0, 1)
        # tof = tof.permute(2, 0, 1)
        # mask = mask.permute(2, 0, 1)

        # 可选的变换
        if self.transform and self.train:
            t1, t1c, t2, tof, mask, seg_result = self.transform((t1, t1c, t2, tof, mask, seg_result))
            # print(mask.shape)
            # print(t1.shape)

        return t1, t1c, t2, tof, mask, data_dict['class'], seg_result, data_dict["name"]

if __name__ == "__main__":
    # 定义数据集路径
    xlsx_dir = '../data/CarotidData/class_label.xlsx'
    txt_file = '../data/CarotidData/train.txt'
    images_dir = '../data/CarotidData/images'
    masks_dir = '../data/CarotidData/masks'
    
    # 创建自定义的 transform
    transform = CustomTransform(normalize=True, flip=True)

    # 创建数据集实例
    dataset = CARO_datasets_Xlsx(xlsx_dir, txt_file, images_dir, masks_dir)

    # 创建DataLoader
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 遍历DataLoader
    for batch_idx, (t1, t1c, t2, tof, mask, cls, seg_result, name) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"T1 shape: {t1.shape}")
        print(f"T1C shape: {t1c.shape}")
        print(f"T2 shape: {t2.shape}")
        print(f"TOF shape: {tof.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"seg_result shape:{seg_result.shape}")
        print("-" * 40)