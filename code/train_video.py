"""
视频疼痛检测训练脚本
- ResNet18 + LSTM 模型
- 支持多模态 (RGB/Depth/Thermal)
- 支持交叉熵和 CORAL 有序回归损失
- 完整训练循环、验证、早停、模型保存
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from dataset_video import VideoFrameSequenceDataset, MultiModalVideoDataset, get_default_transform, get_val_transform
from model_video import VideoPainModel, CoralLoss, labels_to_levels, logits_to_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Train Video Pain Detection Model')
    
    # 数据路径
    parser.add_argument('--data-root', type=str, 
                        default='/home/gm/Workspace/ai-projects/pain_detection/data/mintpain',
                        help='Data root directory')
    parser.add_argument('--train-rgb', type=str, default='train_rgb.txt', help='Train RGB txt file')
    parser.add_argument('--train-depth', type=str, default='train_depth.txt', help='Train Depth txt file')
    parser.add_argument('--train-thermal', type=str, default='train_thermal.txt', help='Train Thermal txt file')
    parser.add_argument('--val-rgb', type=str, default='val_rgb.txt', help='Val RGB txt file')
    parser.add_argument('--val-depth', type=str, default='val_depth.txt', help='Val Depth txt file')
    parser.add_argument('--val-thermal', type=str, default='val_thermal.txt', help='Val Thermal txt file')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=5, help='Number of pain classes (0-4)')
    parser.add_argument('--modalities', type=str, nargs='+', default=['rgb'],
                        help='Input modalities: rgb, depth, thermal')
    parser.add_argument('--pretrained', type=int, default=1, help='Use pretrained ResNet18')
    parser.add_argument('--freeze-backbone', type=int, default=0, help='Freeze ResNet backbone')
    parser.add_argument('--lstm-hidden', type=int, default=256, help='LSTM hidden dimension')
    parser.add_argument('--lstm-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--bidirectional', type=int, default=1, help='Use bidirectional LSTM')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--seq-len', type=int, default=16, help='Sequence length (frames per video)')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # 损失函数
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'coral'],
                        help='Loss type: cross entropy or coral ordinal')
    parser.add_argument('--use-weighted-sampler', type=int, default=1, help='Use weighted sampler')
    parser.add_argument('--use-focal-loss', type=int, default=0, help='Use focal loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    
    # 其他
    parser.add_argument('--save-dir', type=str, default='checkpoints_video',
                        help='Directory to save checkpoints')
    parser.add_argument('--early-stop-patience', type=int, default=10, help='Early stop patience')
    parser.add_argument('--scheduler-patience', type=int, default=5, help='LR scheduler patience')
    parser.add_argument('--min-epochs', type=int, default=10, help='Minimum epochs before early stop')
    parser.add_argument('--sample-mode', type=str, default='uniform', choices=['uniform', 'random'],
                        help='Frame sampling mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset, num_classes):
    """计算类别权重"""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=num_classes)
    safe_counts = np.maximum(counts, 1)
    weights = len(labels) / (num_classes * safe_counts)
    
    return labels, counts, torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal.mean()


def train_epoch(model, loader, criterion, optimizer, device, loss_type, num_classes):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        if isinstance(batch[0], dict):
            # 多模态
            frames_dict, labels = batch
            frames_dict = {k: v.to(device) for k, v in frames_dict.items()}
        else:
            # 单模态
            frames, labels = batch
            frames = frames.to(device)
            frames_dict = {'rgb': frames}
        
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(frames_dict)
        
        if loss_type == 'coral':
            levels = labels_to_levels(labels, num_classes).to(device)
            loss = criterion(logits, levels)
            preds = logits_to_predictions(logits, loss_type, num_classes)
        else:
            loss = criterion(logits, labels)
            preds = logits_to_predictions(logits, loss_type, num_classes)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    return avg_loss, acc, f1, kappa


def validate(model, loader, criterion, device, loss_type, num_classes):
    """验证"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for batch in pbar:
            if isinstance(batch[0], dict):
                frames_dict, labels = batch
                frames_dict = {k: v.to(device) for k, v in frames_dict.items()}
            else:
                frames, labels = batch
                frames = frames.to(device)
                frames_dict = {'rgb': frames}
            
            labels = labels.to(device)
            
            logits = model(frames_dict)
            
            if loss_type == 'coral':
                levels = labels_to_levels(labels, num_classes).to(device)
                loss = criterion(logits, levels)
                preds = logits_to_predictions(logits, loss_type, num_classes)
            else:
                loss = criterion(logits, labels)
                preds = logits_to_predictions(logits, loss_type, num_classes)
            
            total_loss += loss.item() * len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, f1, kappa, cm


def main():
    args = parse_args()
    
    # 设置随机种子
    import random
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 构建数据路径
    data_root = args.data_root
    
    if len(args.modalities) == 1:
        # 单模态
        mod = args.modalities[0]
        train_txt = os.path.join(data_root, getattr(args, f'train_{mod}'))
        val_txt = os.path.join(data_root, getattr(args, f'val_{mod}'))
        
        train_transform = get_default_transform(mod, args.img_size)
        val_transform = get_val_transform(mod, args.img_size)
        
        train_dataset = VideoFrameSequenceDataset(
            txt_path=train_txt,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            transform=train_transform,
            modality=mod,
            sample_mode=args.sample_mode,
        )
        
        val_dataset = VideoFrameSequenceDataset(
            txt_path=val_txt,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            transform=val_transform,
            modality=mod,
            sample_mode=args.sample_mode,
        )
    else:
        # 多模态
        train_txt_paths = {}
        val_txt_paths = {}
        for mod in args.modalities:
            train_txt_paths[mod] = os.path.join(data_root, getattr(args, f'train_{mod}'))
            val_txt_paths[mod] = os.path.join(data_root, getattr(args, f'val_{mod}'))
        
        train_transform = get_default_transform('rgb', args.img_size)
        val_transform = get_val_transform('rgb', args.img_size)
        
        train_dataset = MultiModalVideoDataset(
            txt_paths=train_txt_paths,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            transform=train_transform,
            modalities=args.modalities,
            sample_mode=args.sample_mode,
        )
        
        val_dataset = MultiModalVideoDataset(
            txt_paths=val_txt_paths,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            transform=val_transform,
            modalities=args.modalities,
            sample_mode=args.sample_mode,
        )
    
    print(f"Train dataset: {len(train_dataset)} videos")
    print(f"Val dataset: {len(val_dataset)} videos")
    
    # 计算类别权重
    labels, class_counts, class_weights = compute_class_weights(train_dataset, args.num_classes)
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights: {[round(w, 4) for w in class_weights.tolist()]}")
    
    # 创建 DataLoader
    train_sampler = None
    if args.use_weighted_sampler and args.loss_type != 'coral':
        labels_tensor = torch.from_numpy(labels).long()
        sample_weights = class_weights[labels_tensor].double()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print("Using WeightedRandomSampler")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 创建模型
    model = VideoPainModel(
        num_classes=args.num_classes,
        modalities=args.modalities,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        lstm_hidden_dim=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        output_mode=args.loss_type,
    ).to(device)
    
    print(f"Model: ResNet18 + LSTM")
    print(f"Modalities: {args.modalities}")
    print(f"Loss type: {args.loss_type}")
    
    # 损失函数
    if args.loss_type == 'coral':
        criterion = CoralLoss()
        print("Using CORAL loss")
    elif args.use_focal_loss:
        focal_alpha = None
        if not args.use_weighted_sampler:
            focal_alpha = class_weights.to(device)
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
        print(f"Using Focal loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using Weighted CrossEntropy loss")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.scheduler_patience,
        factor=0.5,
        verbose=True,
    )
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'train_kappa': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_kappa': [],
    }
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练
        train_loss, train_acc, train_f1, train_kappa = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.loss_type, args.num_classes
        )
        
        # 验证
        val_loss, val_acc, val_f1, val_kappa, val_cm = validate(
            model, val_loader, criterion, device,
            args.loss_type, args.num_classes
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_kappa'].append(train_kappa)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_kappa'].append(val_kappa)
        
        # 打印结果
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Kappa: {train_kappa:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
        print(f"Confusion Matrix:\n{val_cm}")
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_kappa': val_kappa,
                'args': vars(args),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model (epoch {epoch})")
        else:
            patience_counter += 1
        
        # 早停
        if epoch >= args.min_epochs and patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best model at epoch {best_epoch}: Val Loss={best_val_loss:.4f}, Val Acc={best_val_acc:.4f}")
            break
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")
    
    # 保存最终模型
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'args': vars(args),
    }
    torch.save(final_checkpoint, os.path.join(args.save_dir, 'final_model.pth'))
    
    print("\nTraining completed!")
    print(f"Best model at epoch {best_epoch}:")
    print(f"  Val Loss: {best_val_loss:.4f}")
    print(f"  Val Acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()