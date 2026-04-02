# 疼痛检测模型 (Pain Detection)

基于 Video Swin Transformer 的产痛评估模型，使用 NRS（Numerical Rating Scale）4分类进行疼痛等级预测。

## 项目简介

本项目利用深度学习技术，通过分析视频片段自动评估疼痛等级。采用预训练的 Video Swin Transformer 作为骨干网络，结合 RetinaFace 人脸检测技术，实现对产痛的精准评估。

### 主要特性

- 🎯 **NRS 4分类**：将疼痛等级分为 0-3 四个级别
- 🧠 **Video Swin Transformer**：使用视频专用 Transformer 架构提取时空特征
- 👤 **RetinaFace 人脸检测**：精准定位和裁剪人脸区域
- 📊 **多指标评估**：支持准确率、F1分数、Cohen's Kappa等多种评估指标
- 🐳 **Docker支持**：提供完整的容器化环境

## 目录结构

```
pain_detection/
├── Dockerfile                 # Docker 环境配置
├── README.md                  # 项目说明文档
├── code/
│   ├── datebase.py            # 数据集加载与预处理
│   ├── model.py               # Video Swin Transformer 模型定义
│   └── train.py               # 训练脚本
├── data/
│   └── videos/
│       ├── labels.csv         # 标签文件
│       ├── train/             # 训练集视频
│       └── val/               # 验证集视频
└── checkpoints/               # 模型保存目录（训练时自动创建）
```

## 环境要求

### 硬件要求
- GPU: 建议 8GB+ 显存（支持 CUDA 12.4）
- 内存: 16GB+

### 软件依赖
- Python 3.10+
- PyTorch 2.5.1+
- CUDA 12.4

## 快速开始

### 1. 使用 Docker（推荐）

```bash
# 构建镜像
docker build -t pain-detection:latest .

# 运行容器（需要 GPU 支持）
docker run --gpus all -it --rm \
    -v $(pwd):/pain_detection \
    pain-detection:latest
```

### 2. 本地环境安装

```bash
# 安装依赖
pip install torch torchvision pytorchvideo opencv-python pandas scikit-learn timm pillow insightface onnxruntime-gpu
```

## 数据准备

### 标签文件格式

创建 `data/train_labels.csv` 和 `data/val_labels.csv`，格式如下：

```csv
video_path,nrs_label
train/video_001.mp4,0
train/video_002.mp4,1
train/video_003.mp4,2
train/video_004.mp4,3
...
```

### NRS 疼痛等级说明

| 等级 | 描述 |
|------|------|
| 0 | 无痛 |
| 1 | 轻度疼痛 |
| 2 | 中度疼痛 |
| 3 | 重度疼痛 |

## 模型训练

### 训练配置

在 `code/train.py` 中的 `Config` 类可以调整以下参数：

```python
class Config:
    # 路径配置
    TRAIN_CSV = "../data/train_labels.csv"
    VAL_CSV = "../data/val_labels.csv"
    SAVE_DIR = "../checkpoints"
    
    # 训练参数
    BATCH_SIZE = 4          # 根据显存调整
    NUM_EPOCHS = 50         # 训练轮数
    LEARNING_RATE = 1e-4    # 学习率
    NUM_WORKERS = 4         # 数据加载线程数
    
    # 模型参数
    NUM_FRAMES = 16         # 每段视频抽帧数
    IMG_SIZE = 224          # 输入图像尺寸
    NUM_CLASSES = 4         # 分类数
    FREEZE_BACKBONE = True  # 是否冻结骨干网络
```

### 启动训练

```bash
cd code
python train.py
```

### 训练输出

训练过程中会输出以下指标：
- **Loss**: 损失值
- **Acc**: 准确率
- **F1**: 加权 F1 分数
- **Kappa**: Cohen's Kappa 系数（医疗任务核心指标）
- **Confusion Matrix**: 混淆矩阵

最佳模型会根据验证集 Kappa 系数自动保存到 `checkpoints/best_model.pth`。

## 模型架构

### VideoSwinPainModel

```
输入: [Batch, 3, 16, 224, 224]
  ↓
Video Swin Tiny (预训练骨干网络)
  ↓
特征向量: [Batch, 768]
  ↓
分类头 (Linear 768→512 → GELU → Dropout → Linear 512→4)
  ↓
输出: [Batch, 4] (NRS 4分类概率)
```

### 关键技术

1. **Video Swin Transformer**: 视频专用 Transformer，有效捕获时空特征
2. **RetinaFace 人脸检测**: 精准定位人脸区域，减少背景干扰
3. **迁移学习**: 使用 Kinetics-400 预训练权重，加速收敛
4. **数据增强**: 随机水平翻转、颜色抖动等，提升泛化能力

## 数据预处理

### 人脸检测与裁剪

项目使用 RetinaFace 进行人脸检测，确保模型聚焦于面部表情：

```python
from datebase import retinaface_crop
import cv2

frame = cv2.imread("video_frame.jpg")
face_crop = retinaface_crop(frame, expand_ratio=0.2)
```

### 视频采样策略

- 均匀采样：从视频中均匀抽取 16 帧
- 时序覆盖：确保覆盖整个视频时长
- 容错处理：检测失败时使用中心裁剪

## 评估指标

### 医疗任务核心指标

- **Cohen's Kappa**: 衡量评估者间一致性，范围 [-1, 1]，越接近 1 越好
- **F1 Score**: 精确率和召回率的调和平均
- **Confusion Matrix**: 展示各类别预测情况

### 指标解读

| Kappa 值 | 一致性强度 |
|----------|-----------|
| < 0.20 | 极低 |
| 0.21-0.40 | 一般 |
| 0.41-0.60 | 中等 |
| 0.61-0.80 | 强 |
| 0.81-1.00 | 极强 |

## 模型推理

```python
import torch
from model import VideoSwinPainModel

# 加载模型
model = VideoSwinPainModel(num_classes=4, pretrained=False)
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    # video_tensor: [1, 3, 16, 224, 224]
    output = model(video_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
```

## 注意事项

1. **显存优化**: 如果显存不足，可以减小 `BATCH_SIZE` 或 `NUM_FRAMES`
2. **数据集大小**: 小数据集建议设置 `FREEZE_BACKBONE=True`
3. **学习率调整**: 使用 ReduceLROnPlateau 自动调整学习率
4. **模型保存**: 基于 Kappa 系数保存最佳模型

## 依赖库

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PyTorchVideo](https://pytorchvideo.org/) - 视频理解库
- [InsightFace](https://github.com/deepinsight/insightface) - 人脸检测
- [OpenCV](https://opencv.org/) - 图像处理
- [scikit-learn](https://scikit-learn.org/) - 评估指标

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请提交 Issue。