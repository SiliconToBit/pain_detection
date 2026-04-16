"""
ResNet18 + LSTM 视频疼痛检测模型
- ResNet18 提取单帧特征
- LSTM 建模时序信息
- 支持多模态输入 (RGB/Depth/Thermal)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):
    """ResNet18 特征提取器，输出 512 维特征"""
    
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        # 加载预训练 ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 移除最后的全连接层，保留特征提取部分
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        # 冻结 backbone
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.out_dim = 512  # ResNet18 layer4 输出通道数
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 单帧图像
        Returns:
            feat: [B, 512] 特征向量
        """
        feat = self.features(x)  # [B, 512, H', W']
        feat = feat.mean(dim=[2, 3])  # Global Average Pooling -> [B, 512]
        return feat


class VideoPainModel(nn.Module):
    """
    ResNet18 + LSTM 视频疼痛检测模型
    
    Args:
        num_classes: 分类数 (默认 5: 0-4 级疼痛)
        modalities: 输入模态列表 ['rgb', 'depth', 'thermal']
        pretrained: 是否使用预训练 ResNet18
        freeze_backbone: 是否冻结 ResNet backbone
        lstm_hidden_dim: LSTM 隐藏层维度
        lstm_num_layers: LSTM 层数
        bidirectional: 是否使用双向 LSTM
        dropout: dropout 比率
        output_mode: 'ce' (分类) 或 'coral' (有序回归)
    """
    
    def __init__(
        self,
        num_classes=5,
        modalities=['rgb'],
        pretrained=True,
        freeze_backbone=False,
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        bidirectional=True,
        dropout=0.5,
        output_mode='ce',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.modalities = modalities
        self.output_mode = output_mode.lower()
        self.bidirectional = bidirectional
        
        # 输出维度
        out_dim = num_classes if self.output_mode == 'ce' else (num_classes - 1)
        
        # 为每个模态创建独立的 ResNet 特征提取器
        self.feature_extractors = nn.ModuleDict()
        for mod in modalities:
            self.feature_extractors[mod] = ResNetFeatureExtractor(
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
        
        # 特征融合后的维度
        feat_dim = 512 * len(modalities)
        
        # LSTM 时序建模
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )
        
        lstm_out_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, out_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入字典或张量
                - 如果是字典: {'rgb': [B, T, C, H, W], 'depth': [B, T, C, H, W], ...}
                - 如果是张量: [B, T, C, H, W] (单模态)
        Returns:
            logits: [B, num_classes] 或 [B, num_classes-1] (coral)
        """
        # 处理输入格式
        if isinstance(x, dict):
            modalities_data = x
        else:
            # 单模态输入，假设是 RGB
            modalities_data = {'rgb': x}
        
        batch_size = None
        seq_len = None
        frame_features = []
        
        # 提取每个模态的帧特征
        for mod in self.modalities:
            if mod not in modalities_data:
                continue
            
            frames = modalities_data[mod]  # [B, T, C, H, W]
            if batch_size is None:
                batch_size = frames.size(0)
                seq_len = frames.size(1)
            
            # 合并 batch 和 time 维度，批量提取特征
            frames_flat = frames.view(-1, frames.size(2), frames.size(3), frames.size(4))  # [B*T, C, H, W]
            feat_flat = self.feature_extractors[mod](frames_flat)  # [B*T, 512]
            
            # 恢复时序结构
            feat_seq = feat_flat.view(batch_size, seq_len, -1)  # [B, T, 512]
            frame_features.append(feat_seq)
        
        # 拼接多模态特征
        if len(frame_features) > 1:
            combined_feat = torch.cat(frame_features, dim=-1)  # [B, T, 512*M]
        else:
            combined_feat = frame_features[0]  # [B, T, 512]
        
        # LSTM 时序建模
        lstm_out, _ = self.lstm(combined_feat)  # [B, T, hidden*2] 或 [B, T, hidden]
        
        # 取最后时刻的输出
        if self.bidirectional:
            # 双向 LSTM: 取正向最后时刻 + 反向第一时刻
            hidden_dim = lstm_out.size(-1) // 2
            last_feat = torch.cat([
                lstm_out[:, -1, :hidden_dim],  # 正向最后
                lstm_out[:, 0, hidden_dim:],   # 反向第一
            ], dim=-1)
        else:
            last_feat = lstm_out[:, -1, :]  # [B, hidden]
        
        # 分类
        logits = self.classifier(last_feat)
        
        return logits
    
    def extract_frame_features(self, x):
        """
        仅提取帧特征，不进行时序建模（用于特征缓存等场景）
        
        Args:
            x: [B, T, C, H, W] 或字典
        Returns:
            feat: [B, T, 512*M]
        """
        if isinstance(x, dict):
            modalities_data = x
        else:
            modalities_data = {'rgb': x}
        
        batch_size = None
        seq_len = None
        frame_features = []
        
        for mod in self.modalities:
            if mod not in modalities_data:
                continue
            
            frames = modalities_data[mod]
            if batch_size is None:
                batch_size = frames.size(0)
                seq_len = frames.size(1)
            
            frames_flat = frames.view(-1, frames.size(2), frames.size(3), frames.size(4))
            feat_flat = self.feature_extractors[mod](frames_flat)
            feat_seq = feat_flat.view(batch_size, seq_len, -1)
            frame_features.append(feat_seq)
        
        if len(frame_features) > 1:
            return torch.cat(frame_features, dim=-1)
        return frame_features[0]


class CoralLoss(nn.Module):
    """CORAL 有序回归损失"""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, levels):
        return self.bce(logits, levels)


def labels_to_levels(labels, num_classes):
    """将类别标签转换为 CORAL 的层级标签"""
    levels = []
    for k in range(num_classes - 1):
        levels.append((labels > k).float())
    return torch.stack(levels, dim=1)


def logits_to_predictions(logits, output_mode, num_classes):
    """将模型输出转换为预测类别"""
    if output_mode == 'coral':
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)
    return torch.argmax(logits, dim=1)


if __name__ == '__main__':
    # 测试模型
    model = VideoPainModel(
        num_classes=5,
        modalities=['rgb'],
        pretrained=False,
        lstm_hidden_dim=256,
        bidirectional=True,
    )
    
    # 模拟输入 [B, T, C, H, W]
    x = torch.randn(2, 16, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 测试多模态
    model_multi = VideoPainModel(
        num_classes=5,
        modalities=['rgb', 'thermal'],
        pretrained=False,
    )
    
    x_multi = {
        'rgb': torch.randn(2, 16, 3, 224, 224),
        'thermal': torch.randn(2, 16, 1, 224, 224),
    }
    output_multi = model_multi(x_multi)
    print(f"Multi-modal output shape: {output_multi.shape}")