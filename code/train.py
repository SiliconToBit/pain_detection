import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import numpy as np

# -------------------------- 修改 1：导入新的模块 --------------------------
from datebase import PainHRDataset  # 替换原来的 PainVideoDataset
from model import HRSingleModalModel  # 替换原来的 VideoSwinPainModel

# -----------------------------------------------------------------------------
# 1. 配置参数（修改部分参数）
# -----------------------------------------------------------------------------
class Config:
    # 路径配置
    TRAIN_CSV = "data/hr/train_hr.csv"  # 改成心率数据 CSV
    VAL_CSV = "data/hr/val_hr.csv"      # 改成心率数据 CSV
    SAVE_DIR = os.getenv("SAVE_DIR", "checkpoints_hr")
    
    # 训练参数
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "100"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-4"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
    EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "12"))
    MIN_EPOCHS = int(os.getenv("MIN_EPOCHS", "15"))
    SCHEDULER_PATIENCE = int(os.getenv("SCHEDULER_PATIENCE", "6"))
    USE_WEIGHTED_SAMPLER = os.getenv("USE_WEIGHTED_SAMPLER", "1") == "1"
    USE_FOCAL_LOSS = os.getenv("USE_FOCAL_LOSS", "1") == "1"
    FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", "2.0"))
    LOSS_TYPE = os.getenv("LOSS_TYPE", "ce").lower()  # ce | coral
    MODEL_ARCH = os.getenv("MODEL_ARCH", "causal_gru").lower()  # causal_gru | legacy_bilstm
    HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "64"))
    
    # 模型参数
    SEQ_LEN = 30             # 心率序列长度（和你 CSV 里的一致）
    NUM_CLASSES = 3          # NRS 3分类（原始数据1-8映射为1,2,3）
    FEATURE_MODE = os.getenv("FEATURE_MODE", "basic").lower()  # basic | enhanced

# 创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)

# 让数据集与评估脚本保持一致的特征配置
os.environ["FEATURE_MODE"] = Config.FEATURE_MODE

# -------------------------- 修改 2：准备心率数据 --------------------------
print("正在加载心率数据集...")
train_dataset = PainHRDataset(
    csv_path=Config.TRAIN_CSV,
    split='train',
    seq_len=Config.SEQ_LEN
)
val_dataset = PainHRDataset(
    csv_path=Config.VAL_CSV,
    split='val',
    seq_len=Config.SEQ_LEN
)

print(f"数据集加载完成：训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")
print(f"归一化模式: {train_dataset.normalize_mode}")
print(f"特征模式: {train_dataset.feature_mode} (channels={train_dataset.in_channels})")

# -------------------------- 修改 3：初始化心率模型 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = HRSingleModalModel(
    seq_len=Config.SEQ_LEN,
    num_classes=Config.NUM_CLASSES,
    in_channels=train_dataset.in_channels,
    output_mode=Config.LOSS_TYPE,
    model_arch=Config.MODEL_ARCH,
    hidden_dim=Config.HIDDEN_DIM,
).to(device)
print(f"模型架构: {Config.MODEL_ARCH}")

# 损失函数、优化器、学习率调度器
def compute_class_weights(dataset, num_classes):
    mapped = dataset.df["nrs_label"].apply(PainHRDataset.map_nrs_to_class).to_numpy(dtype=np.int64)
    counts = np.bincount(mapped, minlength=num_classes)
    safe_counts = np.maximum(counts, 1)
    weights = mapped.shape[0] / (num_classes * safe_counts)
    return mapped, counts, torch.tensor(weights, dtype=torch.float32)


def maybe_override_class_weights(class_weights, num_classes):
    raw = os.getenv("CLASS_WEIGHTS", "").strip()
    if not raw:
        return class_weights, False

    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != num_classes:
        raise ValueError(
            f"CLASS_WEIGHTS expects {num_classes} values, got {len(values)}"
        )
    return torch.tensor(values, dtype=torch.float32), True


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal.mean()


class CoralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, levels):
        return self.bce(logits, levels)


def labels_to_levels(labels, num_classes):
    """将标签 0..K-1 转为 CORAL 的 K-1 阶层标签。"""
    levels = []
    for k in range(num_classes - 1):
        levels.append((labels > k).float())
    return torch.stack(levels, dim=1)


def logits_to_predictions(logits, loss_type):
    if loss_type == "coral":
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)
    return torch.argmax(logits, dim=1)


mapped_labels, class_counts, class_weights = compute_class_weights(train_dataset, Config.NUM_CLASSES)
class_weights, weights_overridden = maybe_override_class_weights(class_weights, Config.NUM_CLASSES)
print(f"训练集类别计数(0/1/2): {class_counts.tolist()}")
print(f"训练集类别权重(0/1/2): {[round(float(x), 4) for x in class_weights.tolist()]}")
if weights_overridden:
    print("类别权重来源: CLASS_WEIGHTS(手动设置)")
else:
    print("类别权重来源: 训练集逆频率")

train_sampler = None
if Config.USE_WEIGHTED_SAMPLER and Config.LOSS_TYPE != "coral":
    mapped_tensor = torch.from_numpy(mapped_labels).long()
    sample_weights = class_weights[mapped_tensor].double()
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print("训练采样策略: WeightedRandomSampler")
else:
    print("训练采样策略: Shuffle")

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=Config.NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

if Config.LOSS_TYPE == "coral":
    criterion = CoralLoss()
    print("损失函数: CORAL (ordinal)")
elif Config.USE_FOCAL_LOSS:
    focal_alpha = None
    if not Config.USE_WEIGHTED_SAMPLER:
        focal_alpha = class_weights.to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=Config.FOCAL_GAMMA)
    if focal_alpha is None:
        print(f"损失函数: FocalLoss(gamma={Config.FOCAL_GAMMA}, alpha=None)")
    else:
        print(f"损失函数: FocalLoss(gamma={Config.FOCAL_GAMMA}, alpha=class_weights)")
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print("损失函数: Weighted CrossEntropy")

print(f"输出模式: {Config.LOSS_TYPE}")

optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=Config.SCHEDULER_PATIENCE, factor=0.5
)

# -----------------------------------------------------------------------------
# 4. 训练 & 验证函数（完全不变）
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for hr_data, labels in loader:
        hr_data, labels = hr_data.to(device), labels.to(device)
        outputs = model(hr_data)
        if Config.LOSS_TYPE == "coral":
            levels = labels_to_levels(labels, Config.NUM_CLASSES)
            loss = criterion(outputs, levels)
        else:
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * hr_data.size(0)
        preds = logits_to_predictions(outputs, Config.LOSS_TYPE).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    return avg_loss, acc, f1, kappa

def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for hr_data, labels in loader:
            hr_data, labels = hr_data.to(device), labels.to(device)
            outputs = model(hr_data)
            if Config.LOSS_TYPE == "coral":
                levels = labels_to_levels(labels, Config.NUM_CLASSES)
                loss = criterion(outputs, levels)
            else:
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * hr_data.size(0)
            preds = logits_to_predictions(outputs, Config.LOSS_TYPE).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return avg_loss, acc, f1, kappa, cm

# -----------------------------------------------------------------------------
# 5. 主训练循环（完全不变）
# -----------------------------------------------------------------------------
best_kappa = -1.0
epochs_no_improve = 0

print("开始训练...")
for epoch in range(Config.NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
    print("-" * 50)
    
    train_loss, train_acc, train_f1, train_kappa = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Kappa: {train_kappa:.4f}")
    
    val_loss, val_acc, val_f1, val_kappa, val_cm = validate(
        model, val_loader, criterion, device, Config.NUM_CLASSES
    )
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
    print("Confusion Matrix:")
    print(val_cm)
    
    scheduler.step(val_loss)
    
    if val_kappa > best_kappa:
        best_kappa = val_kappa
        epochs_no_improve = 0
        save_path = os.path.join(Config.SAVE_DIR, "best_hr_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_kappa': best_kappa,
            'loss_type': Config.LOSS_TYPE,
            'model_arch': Config.MODEL_ARCH,
            'hidden_dim': Config.HIDDEN_DIM,
            'feature_mode': Config.FEATURE_MODE,
            'normalize_mode': train_dataset.normalize_mode,
            'in_channels': train_dataset.in_channels,
            'num_classes': Config.NUM_CLASSES,
        }, save_path)
        print(f"最佳模型已保存至: {save_path} (Kappa: {best_kappa:.4f})")
    else:
        epochs_no_improve += 1

    print(f"EarlyStop 监控: {epochs_no_improve}/{Config.EARLY_STOP_PATIENCE}")
    if (epoch + 1) >= Config.MIN_EPOCHS and epochs_no_improve >= Config.EARLY_STOP_PATIENCE:
        print(
            f"\n触发早停：验证 Kappa 连续 {Config.EARLY_STOP_PATIENCE} 轮未提升，"
            f"在第 {epoch + 1} 轮停止训练。"
        )
        break

print("\n训练完成！")
print(f"最佳验证 Kappa: {best_kappa:.4f}")