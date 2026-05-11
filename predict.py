# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project - Prediction Script

"""
=============================================================================
CMENet 单文件预测脚本
输入 T1, T1C, TOF 三模态 NIfTI 文件，输出分割结果。

用法示例:
    python predict.py \
        --t1 /path/to/t1.nii \
        --t1c /path/to/t1c.nii \
        --tof /path/to/tof.nii \
        -o ./outputs \
        --weight "results/A compare_result/CMENet_NoK_caro_network4_NK_seg/checkpoints/best-epoch1319-loss0.8286.pth"

关键说明:
    输入 nii 文件形状无严格要求，脚本会自动将通道数调整为 30
    (不足则补零，多余则截断) 并将空间尺寸 resize 到 128x128。
=============================================================================
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

from configs.K_config_setting import setting_config
from models.Model.main_models.network import CMENet



# ── 可用的归一化统计量 ──────────────────────────────────────────────
# 键与 config 中 datasets 字段的值对应，"train" 指训练集统计量，
# "test" 指原始 caro 数据集中测试集的统计量。
NORM_STATS = {
    # caro (训练集 / 测试集)
    "caro": {
        "train": {
            "t1_mean": 725.0311808419307, "t1_std": 414.52469564960836,
            "t1c_mean": 1025.7849748164513, "t1c_std": 582.4112496216736,
            "tof_mean": 90.21018866611392, "tof_std": 63.18413996106856,
        },
        "test": {
            "t1_mean": 708.553682879672, "t1_std": 406.7009108433941,
            "t1c_mean": 1001.2429825412771, "t1c_std": 580.6942490142748,
            "tof_mean": 89.5323895397082, "tof_std": 63.63661500556477,
        },
    },
    # new_caro / final
    "new_caro": {
        "train": {
            "t1_mean": 228.4063241394839, "t1_std": 219.4507714371974,
            "t1c_mean": 286.16820705066255, "t1c_std": 266.76549874544384,
            "tof_mean": 120.00167597842182, "tof_std": 169.91381313849973,
        },
        "test": {
            "t1_mean": 228.4063241394839, "t1_std": 219.4507714371974,
            "t1c_mean": 286.16820705066255, "t1c_std": 266.76549874544384,
            "tof_mean": 120.00167597842182, "tof_std": 169.91381313849973,
        },
    },
    "final": {  # 与 new_caro 相同
        "train": {
            "t1_mean": 228.4063241394839, "t1_std": 219.4507714371974,
            "t1c_mean": 286.16820705066255, "t1c_std": 266.76549874544384,
            "tof_mean": 120.00167597842182, "tof_std": 169.91381313849973,
        },
        "test": {
            "t1_mean": 228.4063241394839, "t1_std": 219.4507714371974,
            "t1c_mean": 286.16820705066255, "t1c_std": 266.76549874544384,
            "tof_mean": 120.00167597842182, "tof_std": 169.91381313849973,
        },
    },
}


# ── 辅助工具 ────────────────────────────────────────────────────────

def _adjust_to_30_slices(arr, modality_name):
    """
    将输入数组的最后一维 (C) 调整为 30。
    - 不足: 在末尾补零
    - 超出: 取前 30 个切片
    """
    assert arr.ndim == 3, f"{modality_name} 数据必须为 3 维 (H, W, C)，当前维度: {arr.shape}"
    h, w, c = arr.shape
    if c == 30:
        return arr
    if c < 30:
        pad_width = 30 - c
        padded = np.pad(arr, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        print(f"[INFO] {modality_name}: 通道数不足30，已补零 {pad_width} 通道。")
        return padded
    print(f"[WARNING] {modality_name}: 通道数 {c} > 30，将取前 30 个切片。")
    return arr[:, :, :30]


def _load_nifti(path, modality_name):
    """加载 nii 文件并转换为 float32 numpy (H, W, C)，同时返回仿射矩阵。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{modality_name} 文件不存在: {path}")
    nii_obj = nib.load(path)
    arr = nii_obj.get_fdata().astype(np.float32)
    affine = nii_obj.affine
    if arr.ndim == 4:
        arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr[..., 0]
    return arr, affine


def _normalize(arr, mean, std):
    """Z-score + 逐样本 min-max 归一化到 [0, 255] (与 myNormalize 一致)。"""
    norm = (arr - mean) / std
    arr_min, arr_max = norm.min(), norm.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(norm, dtype=np.float32)
    return ((norm - arr_min) / (arr_max - arr_min) * 255.0).astype(np.float32)


def preprocess_single(t1_arr, t1c_arr, tof_arr, norm_cfg):
    """
    对三个模态的 numpy 数组 (H, W, C) 做预处理，
    返回 tensor tuple: (t1, t1c, tof)，形状均为 (1, 30, 128, 128)。
    """
    original_shape = t1_arr.shape[:2]
    t1_arr = _adjust_to_30_slices(t1_arr, "T1")
    t1c_arr = _adjust_to_30_slices(t1c_arr, "T1C")
    tof_arr = _adjust_to_30_slices(tof_arr, "TOF")

    t1_arr = _normalize(t1_arr, norm_cfg["t1_mean"], norm_cfg["t1_std"])
    t1c_arr = _normalize(t1c_arr, norm_cfg["t1c_mean"], norm_cfg["t1c_std"])
    tof_arr = _normalize(tof_arr, norm_cfg["tof_mean"], norm_cfg["tof_std"])

    t1 = torch.from_numpy(t1_arr).permute(2, 0, 1).contiguous()
    t1c = torch.from_numpy(t1c_arr).permute(2, 0, 1).contiguous()
    tof = torch.from_numpy(tof_arr).permute(2, 0, 1).contiguous()

    t1 = TF.resize(t1, [128, 128])
    t1c = TF.resize(t1c, [128, 128])
    tof = TF.resize(tof, [128, 128])

    return t1.unsqueeze(0), t1c.unsqueeze(0), tof.unsqueeze(0), original_shape


# ── 模型加载 ─────────────────────────────────────────────────────────

def build_model(config, weight_path, device):
    """构建 CMENet 模型并加载权重。"""
    print(f"[INFO] 正在加载模型权重: {weight_path}")
    network = CMENet(config=config)
    model = nn.DataParallel(network, device_ids=[0])
    model.to(device)

    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("[INFO] 模型加载完成。")
    return model


# ── 推理 & 保存 ──────────────────────────────────────────────────────

@torch.no_grad()
def infer_single(model, t1, t1c, tof, device):
    """运行推理，返回分割掩码 (numpy, 128x128x30, float32, 0~1)。"""
    t1 = t1.to(device)
    t1c = t1c.to(device)
    tof = tof.to(device)

    # 构造 dummy seg_result (seg 模式下不会被使用)
    dummy_seg = torch.zeros(1, 30, 128, 128, device=device)

    out, _, _, _, _ = model(t1, t1c, tof, dummy_seg, "seg")
    # out shape: (1, 30, 128, 128)
    mask = out.squeeze(0).cpu().numpy()          # (30, 128, 128)
    mask = np.transpose(mask, (1, 2, 0))          # (128, 128, 30)
    return mask.astype(np.float32)


def save_results(mask, output_dir, base_name, affine, original_shape, threshold=0.5):
    """保存预测结果: .nii 文件 + 中间切片 PNG。"""
    os.makedirs(output_dir, exist_ok=True)

    mask_bin = (mask > threshold).astype(np.int16)

    mask_resized = np.zeros((*original_shape, mask_bin.shape[-1]), dtype=np.int16)
    for i in range(mask_bin.shape[-1]):
        slice_2d = torch.from_numpy(mask_bin[:, :, i]).unsqueeze(0).unsqueeze(0).float()
        slice_resized = TF.resize(slice_2d, list(original_shape), interpolation=TF.InterpolationMode.NEAREST)
        mask_resized[:, :, i] = slice_resized.squeeze().numpy().astype(np.int16)

    nii_path = os.path.join(output_dir, f"{base_name}_pred.nii")
    nii_img = nib.Nifti1Image(mask_resized, affine)
    nib.save(nii_img, nii_path)
    print(f"[INFO] 分割结果已保存至: {nii_path}")

    png_path = os.path.join(output_dir, f"{base_name}_pred.png")
    mid = mask_bin.shape[-1] // 2
    fig, ax = plt.subplots(figsize=(256 / 72, 256 / 72))
    ax.imshow(mask_bin[..., mid], cmap='gray')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(png_path, dpi=72)
    plt.close(fig)
    print(f"[INFO] 预览图片已保存至: {png_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CMENet 三模态分割 — 单文件预测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--t1", required=True, help="T1 nii 文件路径")
    parser.add_argument("--t1c", required=True, help="T1C nii 文件路径")
    parser.add_argument("--tof", required=True,  help="TOF nii 文件路径")
    parser.add_argument("-o", "--output", default="./pred_output",
                        help="输出目录 (默认: ./pred_output)")
    parser.add_argument("-w", "--weight",
                        default="results/A compare_result/CMENet_NoK_caro_network4_NK_seg/checkpoints/best-epoch1319-loss0.8286.pth",
                        help="模型权重 .pth 文件路径")
    parser.add_argument("-d", "--device", default="cuda:0",
                        help="推理设备 (默认: cuda:0, 无 GPU 时设为 cpu)")
    parser.add_argument("--norm_dataset", default="caro",
                        choices=["caro", "new_caro", "final"],
                        help="归一化统计量来源数据集 (默认: caro)")
    parser.add_argument("--norm_split", default="test",
                        choices=["train", "test"],
                        help="使用训练集还是测试集的归一化统计量 (默认: test)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二值化阈值 (默认: 0.5)")
    parser.add_argument("--name", default=None,
                        help="输出文件名前缀 (默认从 T1 文件名推断)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    norm_cfg = NORM_STATS[args.norm_dataset][args.norm_split]

    t1_arr, affine = _load_nifti(args.t1, "T1")
    t1c_arr, _ = _load_nifti(args.t1c, "T1C")
    tof_arr, _ = _load_nifti(args.tof, "TOF")

    base_name = args.name or os.path.splitext(os.path.basename(args.t1))[0]
    for suffix in ["_image", "_T1", "T1"]:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break

    t1_t, t1c_t, tof_t, original_shape = preprocess_single(t1_arr, t1c_arr, tof_arr, norm_cfg)

    config = setting_config
    model = build_model(config, args.weight, device)

    print("[INFO] 正在推理...")
    mask = infer_single(model, t1_t, t1c_t, tof_t, device)

    save_results(mask, args.output, base_name, affine, original_shape, args.threshold)

    print("[DONE] 预测完成。")


if __name__ == "__main__":
    main()
