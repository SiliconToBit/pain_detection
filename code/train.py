import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import numpy as np

# 导入我们自己写的模块
from dataset import PainVideoDataset
from model import VideoSwinPainModel

# -----------------------------------------------------------------------------
# 1. 配置参数
# -----------------------------------------------------------------------------
class Config:
    # 路径配置
    TRAIN_CSV = "../data/train_labels.csv"  # 训练集标签文件
    VAL_CSV = "../data/val_labels.csv"      # 验证集标签文件
    SAVE_DIR = "../checkpoints"              # 模型保存路径
    
    # 训练参数
    BATCH_SIZE = 4          # 根据显存调整，8G显存建议4-8
    NUM_EPOCHS = 50         # 训练轮数
    LEARNING_RATE = 1e-4    # 学习率
    NUM_WORKERS = 4         # 数据加载线程数
    
    # 模型参数
    NUM_FRAMES = 16         # 每段视频抽16帧
    IMG_SIZE = 224          # 图像尺寸
    NUM_CLASSES = 4         # NRS 4分类
    FREEZE_BACKBONE = True  # 小数据集建议冻结骨干网络

# 创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. 准备数据
# -----------------------------------------------------------------------------
print("正在加载数据集...")
train_dataset = PainVideoDataset(
    csv_path=Config.TRAIN_CSV,
    split='train',
    num_frames=Config.NUM_FRAMES,
    img_size=Config.IMG_SIZE
)
val_dataset = PainVideoDataset(
    csv_path=Config.VAL_CSV,
    split='val',
    num_frames=Config.NUM_FRAMES,
    img_size=Config.IMG_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True  # 加速GPU传输
)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True
)
print(f"数据集加载完成：训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")

# -----------------------------------------------------------------------------
# 3. 初始化模型、损失函数、优化器
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = VideoSwinPainModel(
    num_classes=Config.NUM_CLASSES,
    pretrained=True,
    freeze_backbone=Config.FREEZE_BACKBONE
).to(device)

# 损失函数：交叉熵（四分类标准损失）
criterion = nn.CrossEntropyLoss()

# 优化器：AdamW（比Adam更稳定）
optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

# 学习率调度器：验证集损失不下降时降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# -----------------------------------------------------------------------------
# 4. 训练 & 验证函数
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录
        total_loss += loss.item() * videos.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    return avg_loss, acc, f1, kappa

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * videos.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, f1, kappa, cm

# -----------------------------------------------------------------------------
# 5. 主训练循环
# -----------------------------------------------------------------------------
best_kappa = 0.0  # 医疗任务优先看 Kappa 系数

print("开始训练...")
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
        model, val_loader, criterion, device
    )
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
    print("Confusion Matrix:")
    print(val_cm)
    
    # 更新学习率
    scheduler.step(val_loss)
    
    # 保存最佳模型（基于 Kappa 系数）
    if val_kappa > best_kappa:
        best_kappa = val_kappa
        save_path = os.path.join(Config.SAVE_DIR, "best_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_kappa': best_kappa,
        }, save_path)
        print(f"最佳模型已保存至: {save_path} (Kappa: {best_kappa:.4f})")

print("\n训练完成！")
print(f"最佳验证 Kappa: {best_kappa:.4f}")