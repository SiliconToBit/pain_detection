"""
CNN+LSTM 视频疼痛识别训练脚本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import numpy as np

from model_video import build_model
from dataset_video import VideoFrameSequenceDataset, get_default_transform, get_val_transform


# -----------------------------------------------------------------------------
# 1. 配置参数
# -----------------------------------------------------------------------------
class Config:
    # 路径配置
    TRAIN_TXT = os.getenv("TRAIN_TXT", "data/mintpain/train_rgb.txt")
    VAL_TXT = os.getenv("VAL_TXT", "data/mintpain/val_rgb.txt")
    SAVE_DIR = os.getenv("SAVE_DIR", "checkpoints_video")
    
    # 数据配置
    SEQ_LEN = int(os.getenv("SEQ_LEN", "16"))        # 视频帧序列长度
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", "5"))  # 疼痛等级分类数
    IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))      # 图像尺寸
    MODALITY = os.getenv("MODALITY", "rgb")           # 输入模态
    
    # 模型配置
    CNN_MODEL = os.getenv("CNN_MODEL", "resnet18")    # CNN 骨干网络
    HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))  # LSTM 隐藏层维度
    NUM_LSTM_LAYERS = int(os.getenv("NUM_LSTM_LAYERS", "2"))
    DROPOUT = float(os.getenv("DROPOUT", "0.5"))
    BIDIRECTIONAL = os.getenv("BIDIRECTIONAL", "0") == "1"
    PRETRAINED = os.getenv("PRETRAINED", "1") == "1"
    
    # 训练参数
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "50"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
    EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "10"))
    MIN_EPOCHS = int(os.getenv("MIN_EPOCHS", "15"))
    USE_WEIGHTED_SAMPLER = os.getenv("USE_WEIGHTED_SAMPLER", "1") == "1"
    
    # 优化器配置
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    SCHEDULER_PATIENCE = int(os.getenv("SCHEDULER_PATIENCE", "5"))

# 创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. 数据集准备
# -----------------------------------------------------------------------------
print("正在加载视频数据集...")

# 训练集变换 (数据增强)
train_transform = get_default_transform(
    modality=Config.MODALITY,
    img_size=Config.IMG_SIZE
)

# 验证集变换 (无增强)
val_transform = get_val_transform(
    modality=Config.MODALITY,
    img_size=Config.IMG_SIZE
)

train_dataset = VideoFrameSequenceDataset(
    txt_path=Config.TRAIN_TXT,
    seq_len=Config.SEQ_LEN,
    num_classes=Config.NUM_CLASSES,
    transform=train_transform,
    modality=Config.MODALITY,
    sample_mode='random',  # 训练时随机采样
)

val_dataset = VideoFrameSequenceDataset(
    txt_path=Config.VAL_TXT,
    seq_len=Config.SEQ_LEN,
    num_classes=Config.NUM_CLASSES,
    transform=val_transform,
    modality=Config.MODALITY,
    sample_mode='uniform',  # 验证时均匀采样
)

print(f"数据集加载完成: 训练集 {len(train_dataset)} 条, 验证集 {len(val_dataset)} 条")


# -----------------------------------------------------------------------------
# 3. 数据加载器
# -----------------------------------------------------------------------------
def compute_sample_weights(dataset):
    """计算样本权重用于平衡类别"""
    labels = [dataset.video_data[vid]['label'] for vid in dataset.video_ids]
    labels = np.array(labels)
    
    # 计算每个类别的样本数
    class_counts = np.bincount(labels, minlength=Config.NUM_CLASSES)
    class_weights = 1.0 / class_counts
    
    # 为每个样本分配权重
    sample_weights = class_weights[labels]
    
    return torch.tensor(sample_weights, dtype=torch.float64)


train_sampler = None
if Config.USE_WEIGHTED_SAMPLER:
    sample_weights = compute_sample_weights(train_dataset)
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print("使用 WeightedRandomSampler 平衡类别")

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=Config.NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# -----------------------------------------------------------------------------
# 4. 模型、损失函数、优化器
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = build_model(
    num_classes=Config.NUM_CLASSES,
    cnn_model=Config.CNN_MODEL,
    hidden_dim=Config.HIDDEN_DIM,
    num_lstm_layers=Config.NUM_LSTM_LAYERS,
    dropout=Config.DROPOUT,
    bidirectional=Config.BIDIRECTIONAL,
    pretrained=Config.PRETRAINED,
).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器 (只优化 LSTM 和分类头,CNN 使用较小的学习率)
optimizer = optim.AdamW([
    {'params': model.cnn.parameters(), 'lr': Config.LEARNING_RATE * 0.1},
    {'params': model.lstm.parameters(), 'lr': Config.LEARNING_RATE},
    {'params': model.classifier.parameters(), 'lr': Config.LEARNING_RATE},
], weight_decay=Config.WEIGHT_DECAY)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=Config.SCHEDULER_PATIENCE, factor=0.5
)


# -----------------------------------------------------------------------------
# 5. 训练 & 验证函数
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * frames.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    return avg_loss, acc, f1, kappa


def validate(model, loader, criterion, device, num_classes):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * frames.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    return avg_loss, acc, f1, kappa, cm


# -----------------------------------------------------------------------------
# 6. 主训练循环
# -----------------------------------------------------------------------------
best_kappa = -1.0
epochs_no_improve = 0

print("\n" + "="*60)
print("开始训练 CNN+LSTM 视频疼痛识别模型")
print("="*60 + "\n")

for epoch in range(Config.NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
    print("-" * 50)
    
    # 训练
    train_loss, train_acc, train_f1, train_kappa = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Kappa: {train_kappa:.4f}")
    
    # 验证
    val_loss, val_acc, val_f1, val_kappa, val_cm = validate(
        model, val_loader, criterion, device, Config.NUM_CLASSES
    )
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
    print("Confusion Matrix:")
    print(val_cm)
    
    # 更新学习率
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr:.6f}")
    
    # 保存最佳模型
    if val_kappa > best_kappa:
        best_kappa = val_kappa
        epochs_no_improve = 0
        
        save_path = os.path.join(Config.SAVE_DIR, "best_video_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_kappa': best_kappa,
            'config': {
                'num_classes': Config.NUM_CLASSES,
                'cnn_model': Config.CNN_MODEL,
                'hidden_dim': Config.HIDDEN_DIM,
                'num_lstm_layers': Config.NUM_LSTM_LAYERS,
                'dropout': Config.DROPOUT,
                'bidirectional': Config.BIDIRECTIONAL,
                'seq_len': Config.SEQ_LEN,
                'img_size': Config.IMG_SIZE,
                'modality': Config.MODALITY,
            },
        }, save_path)
        print(f"✓ 最佳模型已保存: {save_path} (Kappa: {best_kappa:.4f})")
    else:
        epochs_no_improve += 1
    
    # Early Stopping
    print(f"EarlyStop 监控: {epochs_no_improve}/{Config.EARLY_STOP_PATIENCE}")
    if (epoch + 1) >= Config.MIN_EPOCHS and epochs_no_improve >= Config.EARLY_STOP_PATIENCE:
        print(f"\n触发早停: 验证 Kappa 连续 {Config.EARLY_STOP_PATIENCE} 轮未提升")
        break

print("\n" + "="*60)
print("训练完成!")
print(f"最佳验证 Kappa: {best_kappa:.4f}")
print("="*60)
