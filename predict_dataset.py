# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project - Prediction Script

"""
=============================================================================
CMENet 数据集批量预测脚本
给定一个包含 T1 / T1C / TOF 三模态 NIfTI 文件的数据集目录，
批量推理并输出所有样本的分割结果。

======================================================================
 文件命名规则说明 (重要! 请根据你的数据集选择对应模式)
======================================================================

本脚本支持自动识别常见数据集结构和三种文件命名模式。
用户只需要给数据集根目录，脚本会自动查找 T1 / T1C / TOF 三个目录。

支持的数据集目录结构：

  结构 1：根目录下直接有三种模态目录
    dataset_dir/T1
    dataset_dir/T1C
    dataset_dir/TOF

  结构 2：根目录下有 image 目录，image 下有三种模态目录
    dataset_dir/image/T1
    dataset_dir/image/T1C
    dataset_dir/image/TOF

支持三种文件命名模式，通过 --name_pattern 指定；默认 auto 自动识别：

  mode="image" (caro 数据集风格)
    T1/T1C/TOF 三个目录下的文件名完全一致
    ├── T1/patient01_image.nii
    ├── T1C/patient01_image.nii
    └── TOF/patient01_image.nii

  mode="T1" (crop30 / weifang / QZ / SZ 数据集风格)
    每种模态文件名带有模态后缀
    ├── T1/patient01_T1.nii
    ├── T1C/patient01_T1C.nii
    └── TOF/patient01_TOF.nii

  mode="direct" (final / new_caro 数据集风格)
    模态名紧跟在名称后
    ├── T1/patient01T1.nii
    ├── T1C/patient01T1C.nii
    └── TOF/patient01TOF.nii

  如果以上三种都不适用，你也可以手动提供一个 txt 文件
  (每行一个样本名称)，然后用 --txt 参数指定，配合 --name_pattern
  来控制每个模态的文件名构造方式。

======================================================================

用法示例:

  # 自动扫描数据集目录，自动识别 image/T1/direct 命名模式
  python predict_dataset.py \
      -d /path/to/dataset \
      -o ./pred_output \
      --weight "results/A compare_result/CMENet_NoK_caro_network4_NK_seg/checkpoints/best-epoch1319-loss0.8286.pth"

  # 使用 txt 文件指定样本列表
  python predict_dataset.py \
      -d /path/to/dataset/images \
      --txt /path/to/samples.txt \
      -o ./pred_output

=============================================================================
"""

import os
import sys
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from configs.K_config_setting import setting_config
from models.Model.main_models.network import CMENet


# ── 归一化统计量 (与 predict.py 共用) ────────────────────────────────

NORM_STATS = {
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
    "final": {
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


# ── 文件名构造 ───────────────────────────────────────────────────────

def _build_file_name(name, modality, name_pattern):
    """根据样本名称和命名模式构造 nii 文件名（不含扩展名）。"""
    if name_pattern == "image":
        return f"{name}_image"
    elif name_pattern == "T1":
        return f"{name}_{modality.upper()}"
    elif name_pattern == "direct":
        return f"{name}{modality.upper()}"
    else:
        raise ValueError(
            f"不支持的命名模式: {name_pattern}，"
            f"可选: 'image', 'T1', 'direct'"
        )


def resolve_images_dir(dataset_dir):
    """兼容 dataset_dir/T1... 和 dataset_dir/image/T1... 两种结构。"""
    direct_dir = dataset_dir
    nested_dir = os.path.join(dataset_dir, "image")
    for candidate in (direct_dir, nested_dir):
        if all(os.path.isdir(os.path.join(candidate, m)) for m in ("T1", "T1C", "TOF")):
            return candidate
    raise NotADirectoryError(
        "未找到 T1/T1C/TOF 三个模态目录。请确认数据集结构为：\n"
        "  dataset_dir/T1, dataset_dir/T1C, dataset_dir/TOF\n"
        "或：\n"
        "  dataset_dir/image/T1, dataset_dir/image/T1C, dataset_dir/image/TOF"
    )


def resolve_txt_file(dataset_dir, images_dir, txt_file):
    """优先使用用户指定 txt；否则自动使用数据集根目录下的 test.txt。"""
    if txt_file:
        return txt_file
    candidates = [
        os.path.join(dataset_dir, "test.txt"),
        os.path.join(os.path.dirname(images_dir), "test.txt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


# ── 样本发现 ─────────────────────────────────────────────────────────

def discover_samples(images_dir, name_pattern="auto", txt_file=None):
    """
    发现数据集中的所有样本名称。

    Returns:
        list[str]: 样本名称列表
    """
    t1_dir = os.path.join(images_dir, "T1")
    t1c_dir = os.path.join(images_dir, "T1C")
    tof_dir = os.path.join(images_dir, "TOF")

    for d, label in [(t1_dir, "T1"), (t1c_dir, "T1C"), (tof_dir, "TOF")]:
        if not os.path.isdir(d):
            raise NotADirectoryError(f"{label} 目录不存在: {d}")

    patterns = ["image", "T1", "direct"] if name_pattern == "auto" else [name_pattern]

    if txt_file is not None and os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        detected_pattern = _detect_pattern_from_names(names, patterns, t1_dir, t1c_dir, tof_dir)
        if detected_pattern is None:
            raise FileNotFoundError(
                f"txt 中的样本名称无法匹配 T1/T1C/TOF 文件。txt: {txt_file}"
            )
        return names, detected_pattern
    else:
        # 从 T1 目录自动发现
        t1_files = set()
        for f in glob.glob(os.path.join(t1_dir, "*.nii")) + \
                  glob.glob(os.path.join(t1_dir, "*.nii.gz")):
            t1_files.add(os.path.basename(f))

        # 提取样本名称
        names = set()
        detected_pattern = None
        for pattern in patterns:
            current_names = set()
            for fname in t1_files:
                name = _extract_name(fname, pattern)
                if name:
                    # 验证 T1C 和 TOF 对应文件也存在
                    t1c_base = _build_file_name(name, "t1c", pattern)
                    tof_base = _build_file_name(name, "tof", pattern)
                    try:
                        _resolve_nifti_path(t1c_dir, t1c_base)
                        _resolve_nifti_path(tof_dir, tof_base)
                        current_names.add(name)
                    except FileNotFoundError:
                        pass
            if current_names:
                names = current_names
                detected_pattern = pattern
                break

        if not names:
            print("[WARNING] 未能自动发现样本。")
            if name_pattern == "image":
                print("  T1/ 目录下文件应以 _image.nii 结尾 (caro 风格)")
            elif name_pattern == "T1":
                print("  T1/ 目录下文件应以 _T1.nii 结尾 (crop30 风格)")
            elif name_pattern == "direct":
                print("  T1/ 目录下文件应以 T1.nii 结尾 (final 风格)")
            print("  请检查数据集中 T1/T1C/TOF 文件命名是否一致，或使用 --txt 手动指定样本列表。")
            sys.exit(1)

        names = sorted(list(names))

    return names, detected_pattern


def _detect_pattern_from_names(names, patterns, t1_dir, t1c_dir, tof_dir):
    """给定样本名称列表，检测哪种命名模式能同时找到 T1/T1C/TOF。"""
    for pattern in patterns:
        ok = True
        for name in names:
            try:
                _resolve_nifti_path(t1_dir, _build_file_name(name, "t1", pattern))
                _resolve_nifti_path(t1c_dir, _build_file_name(name, "t1c", pattern))
                _resolve_nifti_path(tof_dir, _build_file_name(name, "tof", pattern))
            except FileNotFoundError:
                ok = False
                break
        if ok:
            return pattern
    return None


def _extract_name(fname, name_pattern):
    """从文件名中提取样本名称。"""
    if name_pattern == "image":
        if fname.endswith("_image.nii"):
            return fname[:-len("_image.nii")]
        elif fname.endswith("_image.nii.gz"):
            return fname[:-len("_image.nii.gz")]
    elif name_pattern == "T1":
        if fname.endswith("_T1.nii"):
            return fname[:-len("_T1.nii")]
        elif fname.endswith("_T1.nii.gz"):
            return fname[:-len("_T1.nii.gz")]
    elif name_pattern == "direct":
        if fname.endswith("T1.nii"):
            return fname[:-len("T1.nii")]
        elif fname.endswith("T1.nii.gz"):
            return fname[:-len("T1.nii.gz")]
    return None


# ── 预处理 ────────────────────────────────────────────────────────────

def _adjust_to_30_slices(arr, modality_name):
    assert arr.ndim == 3, f"{modality_name} 数据必须为 3 维，当前维度: {arr.shape}"
    h, w, c = arr.shape
    if c == 30:
        return arr
    if c < 30:
        pad_width = 30 - c
        return np.pad(arr, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
    # 多于 30，居中截取
    start = (c - 30) // 2
    return arr[:, :, start:start + 30]


def _resolve_nifti_path(dir_path, fname_base):
    """在 dir_path 中查找 fname_base(.nii 或 .nii.gz)。"""
    for ext in (".nii", ".nii.gz"):
        candidate = os.path.join(dir_path, fname_base + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"文件不存在: {os.path.join(dir_path, fname_base)}.nii[.gz]")


def _load_nifti(path, modality_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{modality_name} 文件不存在: {path}")
    arr = nib.load(path).get_fdata().astype(np.float32)
    if arr.ndim == 4:
        arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr[..., 0]
    return arr


def _normalize(arr, mean, std):
    norm = (arr - mean) / std
    arr_min, arr_max = norm.min(), norm.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(norm, dtype=np.float32)
    return ((norm - arr_min) / (arr_max - arr_min) * 255.0).astype(np.float32)


def preprocess_single(t1_arr, t1c_arr, tof_arr, norm_cfg):
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

    return t1.unsqueeze(0), t1c.unsqueeze(0), tof.unsqueeze(0)


# ── 模型加载 ─────────────────────────────────────────────────────────

def build_model(config, weight_path, device):
    print(f"[INFO] 正在加载模型权重: {weight_path}")
    network = CMENet(config=config)
    model = nn.DataParallel(network, device_ids=[0])
    model.to(device)

    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("[INFO] 模型加载完成。")
    return model


# ── 推理 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_single(model, t1, t1c, tof, device):
    t1 = t1.to(device)
    t1c = t1c.to(device)
    tof = tof.to(device)
    dummy_seg = torch.zeros(1, 30, 128, 128, device=device)

    out, _, _, _, _ = model(t1, t1c, tof, dummy_seg, "seg")
    mask = out.squeeze(0).cpu().numpy()          # (30, 128, 128)
    mask = np.transpose(mask, (1, 2, 0))          # (128, 128, 30)
    return mask.astype(np.float32)


def save_results(mask, output_dir, name, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    mask_bin = (mask > threshold).astype(np.int16)

    nii_path = os.path.join(output_dir, f"{name}_pred.nii")
    nib.save(nib.Nifti1Image(mask_bin, np.eye(4)), nii_path)

    png_path = os.path.join(output_dir, f"{name}_pred.png")
    mid = mask_bin.shape[-1] // 2
    fig, ax = plt.subplots(figsize=(256 / 72, 256 / 72))
    ax.imshow(mask_bin[..., mid], cmap='gray')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(png_path, dpi=72)
    plt.close(fig)


# ── 批量推理主流程 ───────────────────────────────────────────────────

def process_dataset(images_dir, names, name_pattern, model, norm_cfg, device,
                    output_dir, threshold):
    t1_dir = os.path.join(images_dir, "T1")
    t1c_dir = os.path.join(images_dir, "T1C")
    tof_dir = os.path.join(images_dir, "TOF")

    failed = []
    pbar = tqdm(names, desc="推理进度", unit="样本")
    for name in pbar:
        pbar.set_postfix_str(name)

        t1_base = _build_file_name(name, "t1", name_pattern)
        t1c_base = _build_file_name(name, "t1c", name_pattern)
        tof_base = _build_file_name(name, "tof", name_pattern)

        try:
            t1_path = _resolve_nifti_path(t1_dir, t1_base)
            t1c_path = _resolve_nifti_path(t1c_dir, t1c_base)
            tof_path = _resolve_nifti_path(tof_dir, tof_base)
            t1_arr = _load_nifti(t1_path, "T1")
            t1c_arr = _load_nifti(t1c_path, "T1C")
            tof_arr = _load_nifti(tof_path, "TOF")

            t1_t, t1c_t, tof_t = preprocess_single(t1_arr, t1c_arr, tof_arr, norm_cfg)
            mask = infer_single(model, t1_t, t1c_t, tof_t, device)
            save_results(mask, output_dir, name, threshold)
        except Exception as e:
            print(f"\n[ERROR] 样本 '{name}' 处理失败: {e}")
            failed.append(name)

    return failed


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CMENet 三模态分割 — 数据集批量预测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
文件命名模式说明 (--name_pattern，默认 auto 自动识别):
  auto    : 自动识别 image / T1 / direct
  image   : T1/xxx_image.nii   T1C/xxx_image.nii   TOF/xxx_image.nii    (caro 风格)
  T1      : T1/xxx_T1.nii      T1C/xxx_T1C.nii     TOF/xxx_TOF.nii      (crop30 风格)
  direct  : T1/xxxT1.nii       T1C/xxxT1C.nii      TOF/xxxTOF.nii       (final 风格)

支持的数据集目录结构:
  dataset_dir/
    ├── T1/
    ├── T1C/
    └── TOF/

或:
  dataset_dir/
    └── image/
        ├── T1/
        ├── T1C/
        └── TOF/
        """,
    )
    parser.add_argument("-d", "--dataset_dir", required=True,
                        help="数据集根目录，支持直接包含 T1/T1C/TOF，或包含 image/T1 image/T1C image/TOF")
    parser.add_argument("--name_pattern", default="auto",
                        choices=["auto", "image", "T1", "direct"],
                        help="文件命名模式，默认 auto 自动识别")
    parser.add_argument("--txt", default=None,
                        help="样本名称列表文件 (每行一个名称)。不提供则自动从 T1/ 目录扫描。")
    parser.add_argument("-o", "--output", default="./batch_output",
                        help="输出目录 (默认: ./batch_output)")
    parser.add_argument("-w", "--weight",
                        default="results/A compare_result/CMENet_NoK_caro_network4_NK_seg/checkpoints/best-epoch1319-loss0.8286.pth",
                        help="模型权重 .pth 文件路径")
    parser.add_argument("-D", "--device", default="cuda:0",
                        help="推理设备 (默认: cuda:0)")
    parser.add_argument("--norm_dataset", default="caro",
                        choices=["caro", "new_caro", "final"],
                        help="归一化统计量来源数据集 (默认: caro)")
    parser.add_argument("--norm_split", default="test",
                        choices=["train", "test"],
                        help="使用训练集还是测试集的归一化统计量 (默认: test)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二值化阈值 (默认: 0.5)")

    args = parser.parse_args()

    # ── 检查数据集目录结构 ──
    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] 数据集目录不存在: {dataset_dir}")
        sys.exit(1)
    try:
        images_dir = resolve_images_dir(dataset_dir)
    except NotADirectoryError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    txt_file = resolve_txt_file(dataset_dir, images_dir, args.txt)

    # ── 设备 ──
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ── 发现样本 ──
    print(f"[INFO] 输入数据集根目录: {dataset_dir}")
    print(f"[INFO] 三模态数据目录: {images_dir}")
    print(f"[INFO] 命名模式: {args.name_pattern}")
    if txt_file:
        print(f"[INFO] 样本列表: {txt_file}")
    print(f"[INFO] 正在扫描数据集: {images_dir}")
    names, detected_pattern = discover_samples(images_dir, args.name_pattern, txt_file)
    print(f"[INFO] 实际使用命名模式: {detected_pattern}")
    print(f"[INFO] 发现 {len(names)} 个样本")
    if not names:
        print("[ERROR] 未找到任何样本，退出。")
        sys.exit(1)

    # ── 归一化配置 ──
    norm_cfg = NORM_STATS[args.norm_dataset][args.norm_split]

    # ── 构建模型 ──
    config = setting_config
    model = build_model(config, args.weight, device)

    # ── 批量推理 ──
    print(f"\n[INFO] 开始批量推理...")
    failed = process_dataset(
        images_dir, names, detected_pattern,
        model, norm_cfg, device, args.output, args.threshold
    )

    # ── 汇总 ──
    n_total = len(names)
    n_ok = n_total - len(failed)
    print(f"\n{'='*60}")
    print(f"批量推理完成: {n_ok}/{n_total} 成功")
    if failed:
        print(f"失败样本 ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    print(f"结果保存在: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
