# CMENet

CMENet 是一个面向三模态医学影像数据的分割项目。当前项目支持输入 `T1`、`T1C`、`TOF` 三种 NIfTI 格式数据，并利用训练好的 CMENet 预训练权重输出分割结果。

项目当前重点支持陌生数据标注场景：用户提供单个病例的三模态数据，或提供一个包含三模态数据的新数据集，模型即可自动完成推理并保存分割结果。

## 项目功能

本项目主要完成以下工作：

- 使用 CMENet 网络对三模态医学影像进行分割。
- 支持 `T1`、`T1C`、`TOF` 三种模态作为模型输入。
- 支持加载已训练好的 CMENet 权重进行推理。
- 支持单个病例预测。
- 支持整个数据集批量预测。
- 支持 Windows 系统下通过 `.bat` 脚本直接运行。
- 输出结果包括 `.nii` 分割文件和 `.png` 中间切片预览图。

默认权重路径为：

```text
results/A compare_result/CMENet_NoK_caro_network4_NK_seg/checkpoints/best-epoch1319-loss0.8286.pth
```

## 作者信息

- 傅钊进，北京科技大学人工智能学院博士，目前在中科院自动化所联培。
- 盖群，首都医科大学宣武医院博士。

## 环境配置

推荐使用 Conda 环境。当前项目环境名称为 `CMENet`。

### 1. 创建或进入环境

如果已经有 `CMENet` 环境：

```bat
conda activate CMENet
```

如果需要重新创建环境：

```bat
conda create -n CMENet python=3.9 -y
conda activate CMENet
```

### 2. 安装 PyTorch GPU 版本

如果机器有 NVIDIA GPU，推荐安装 CUDA 11.8 对应版本：

```bat
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 使用清华源安装项目依赖

```bat
pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 matplotlib==3.7.2 nibabel==5.1.0 tqdm==4.65.0 torchmetrics==1.0.0 timm==0.9.2 efficientnet_pytorch==0.7.1 einops mmengine openpyxl opencv-python==4.8.1.78 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

也可以安装 `requirements.txt` 中的主要依赖：

```bat
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

注意：项目中实际还使用了 `einops`、`mmengine`、`openpyxl`，如果使用 `requirements.txt` 后仍提示缺包，请额外安装：

```bat
pip install einops mmengine openpyxl opencv-python==4.8.1.78 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 4. 验证环境

在项目根目录运行：

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果输出中 `torch.cuda.is_available()` 为 `True`，说明 GPU 可用。

也可以验证模型导入：

```bat
python -c "from models.Model.main_models.network import CMENet; print('CMENet import OK')"
```

## 数据格式要求

输入数据应为 NIfTI 文件，支持：

```text
.nii
.nii.gz
```

模型输入包含三种模态：

```text
T1
T1C
TOF
```

预测脚本会自动将数据调整为模型需要的输入形式：

- 切片数调整为 30。
- 空间尺寸 resize 到 `128 x 128`。
- 使用项目中的归一化方式进行预处理。

## 使用方式

项目提供两类预测方式：

- `predict.py` / `predict.bat`：单个病例预测。
- `predict_dataset.py` / `predict_dataset.bat`：整个数据集批量预测。

Windows 用户推荐直接使用 `.bat` 脚本。

## Windows 下运行说明

在 Windows 的 `cmd` 或 Anaconda Prompt 中运行。

首先进入项目目录并激活环境：

```bat
cd /d C:\Users\FZJ\Desktop\CMENet
conda activate CMENet
```

### 单病例预测

运行：

```bat
predict.bat
```

然后按提示输入：

```text
T1 文件路径
T1C 文件路径
TOF 文件路径
输出目录
权重路径
```

也可以直接在命令行中提供参数：

```bat
predict.bat "D:\data\patient01_T1.nii" "D:\data\patient01_T1C.nii" "D:\data\patient01_TOF.nii"
```

指定输出目录：

```bat
predict.bat "D:\data\patient01_T1.nii" "D:\data\patient01_T1C.nii" "D:\data\patient01_TOF.nii" "D:\output"
```

指定输出目录和权重：

```bat
predict.bat "D:\data\patient01_T1.nii" "D:\data\patient01_T1C.nii" "D:\data\patient01_TOF.nii" "D:\output" "D:\weights\best.pth"
```

输出结果默认保存在：

```text
pred_output/
```

输出文件包括：

```text
xxx_pred.nii
xxx_pred.png
```

### 数据集批量预测

运行：

```bat
predict_dataset.bat
```

然后按提示输入数据集目录。

也可以直接提供数据集路径：

```bat
predict_dataset.bat "D:\your_dataset"
```

脚本会自动识别常见数据集结构和文件命名方式。

支持的数据集结构 1：

```text
your_dataset/
├── T1/
├── T1C/
└── TOF/
```

支持的数据集结构 2：

```text
your_dataset/
├── image/
│   ├── T1/
│   ├── T1C/
│   └── TOF/
└── test.txt
```

当前项目中的示例数据 `data/c` 就是第二种结构：

```text
data/c/
├── image/
│   ├── T1/
│   ├── T1C/
│   └── TOF/
├── masks/
├── class_label.xlsx
└── test.txt
```

因此可以直接运行：

```bat
predict_dataset.bat data\c
```

或：

```bat
predict_dataset.bat "C:\Users\FZJ\Desktop\CMENet\data\c"
```

### 数据集文件命名规则

批量预测脚本默认使用 `auto` 模式自动识别文件命名方式。

支持以下三种常见命名方式：

#### 1. image 模式

```text
T1/xxx_image.nii
T1C/xxx_image.nii
TOF/xxx_image.nii
```

示例：

```bat
predict_dataset.bat data\c image
```

#### 2. T1 模式

```text
T1/xxx_T1.nii
T1C/xxx_T1C.nii
TOF/xxx_TOF.nii
```

示例：

```bat
predict_dataset.bat "D:\your_dataset" T1
```

#### 3. direct 模式

```text
T1/xxxT1.nii
T1C/xxxT1C.nii
TOF/xxxTOF.nii
```

示例：

```bat
predict_dataset.bat "D:\your_dataset" direct
```

### Windows 路径注意事项

如果路径中包含空格，必须使用英文双引号：

```bat
predict_dataset.bat "D:\my data\dataset"
```

不建议在 Windows CMD 中运行 `.sh` 文件。如果需要运行 `.sh`，需要安装 Git Bash 或 WSL。但本项目已经提供 Windows 原生 `.bat` 脚本，推荐直接使用 `.bat`。

## Python 脚本直接运行

如果不使用 `.bat`，也可以直接运行 Python 脚本。

### 单病例预测

```bat
python predict.py --t1 "D:\data\T1.nii" --t1c "D:\data\T1C.nii" --tof "D:\data\TOF.nii" -o "D:\output"
```

### 数据集批量预测

```bat
python predict_dataset.py -d "D:\your_dataset" -o "D:\output"
```

指定命名模式：

```bat
python predict_dataset.py -d "D:\your_dataset" --name_pattern T1 -o "D:\output"
```

指定权重：

```bat
python predict_dataset.py -d "D:\your_dataset" -w "D:\weights\best.pth" -o "D:\output"
```

## 输出结果

预测完成后，每个样本会生成：

```text
sample_pred.nii
sample_pred.png
```

其中：

- `.nii` 是二值化后的分割结果。
- `.png` 是中间切片预览图，便于快速查看结果。

## 常见问题

### 1. Windows 下运行 `.sh` 报错

如果出现类似：

```text
适用于 Linux 的 Windows 子系统没有已安装的分发
```

说明当前 `bash` 调用了 WSL，但没有安装 Linux 子系统。Windows 下推荐使用：

```bat
predict.bat
predict_dataset.bat
```

### 2. 路径中有空格导致找不到文件

请使用英文双引号包裹路径：

```bat
"D:\my data\patient01_T1.nii"
```

### 3. 找不到 `T1/T1C/TOF` 目录

请确认数据集目录满足以下任一结构：

```text
dataset/T1
dataset/T1C
dataset/TOF
```

或：

```text
dataset/image/T1
dataset/image/T1C
dataset/image/TOF
```

### 4. 找不到权重文件

请确认默认权重存在：

```text
results/A compare_result/CMENet_NoK_caro_network4_NK_seg/checkpoints/best-epoch1319-loss0.8286.pth
```

如果权重放在其他位置，可以在运行时指定权重路径。
