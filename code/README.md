# CNN+LSTM 视频疼痛识别模型

## 项目结构

```
code/
├── model_video.py      # CNN+LSTM 模型定义
├── dataset_video.py    # 视频数据集加载
├── train_video.py      # 训练脚本
├── predict.py          # 预测脚本
└── generate_data_lists.py  # 生成训练/验证数据列表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成数据列表

```bash
python code/generate_data_lists.py \
    --data_root /path/to/mintpain \
    --output_dir data/mintpain
```

### 3. 训练模型

**本地测试 (CPU):**
```bash
python code/train_video.py
```

**云端训练 (GPU):**
```bash
# 使用环境变量配置
SEQ_LEN=16 \
BATCH_SIZE=8 \
NUM_EPOCHS=50 \
LEARNING_RATE=1e-4 \
CNN_MODEL=resnet18 \
HIDDEN_DIM=256 \
python code/train_video.py
```

**后台运行:**
```bash
nohup python code/train_video.py > train.log 2>&1 &
tail -f train.log
```

### 4. 预测

```bash
python code/predict.py \
    --checkpoint checkpoints_video/best_video_model.pth \
    --video_dir "/path/to/video/frames"
```

## 模型架构

```
视频帧序列 [T, C, H, W]
    ↓
CNN 特征提取 (ResNet18/34/50)
    ↓
帧特征序列 [T, feature_dim]
    ↓
LSTM 时序建模
    ↓
分类头 (全连接层)
    ↓
疼痛等级 [0, 1, 2, 3, 4]
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SEQ_LEN | 16 | 视频帧序列长度 |
| BATCH_SIZE | 8 | 批次大小 |
| NUM_EPOCHS | 50 | 训练轮数 |
| LEARNING_RATE | 1e-4 | 学习率 |
| CNN_MODEL | resnet18 | CNN 骨干网络 |
| HIDDEN_DIM | 256 | LSTM 隐藏层维度 |
| NUM_LSTM_LAYERS | 2 | LSTM 层数 |
| DROPOUT | 0.5 | Dropout 比率 |
| BIDIRECTIONAL | 0 | 是否双向 LSTM |
| MODALITY | rgb | 输入模态 (rgb/depth/thermal) |
| IMG_SIZE | 224 | 图像尺寸 |

## 云端训练工作流

### 本地 (开发)
1. 编写/修改代码
2. Git 提交
3. 推送代码

### 云端 (训练)
1. `git pull` 拉取最新代码
2. `pip install -r requirements.txt` 安装依赖
3. 运行训练脚本
4. 监控训练日志

### 本地 (分析)
1. 下载训练好的模型
2. 使用 predict.py 测试
3. 分析结果

## 注意事项

1. **数据路径**: 确保 txt 文件中的路径在云端服务器上有效
2. **GPU 显存**: 根据显存调整 BATCH_SIZE 和 CNN_MODEL
3. **预训练权重**: 首次运行会自动下载 ResNet 预训练权重
4. **Early Stopping**: 验证 Kappa 连续 10 轮未提升会触发早停
