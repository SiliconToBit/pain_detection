"""
CNN+LSTM 视频疼痛识别模型
- CNN 提取每帧的空间特征
- LSTM 建模时序依赖
- 输出疼痛等级分类
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class CNNFeatureExtractor(nn.Module):
    """
    CNN 特征提取器
    
    使用预训练的 ResNet 提取每帧的特征向量
    移除最后的全连接层,输出特征维度为 512 (ResNet18) 或 2048 (ResNet50)
    """
    
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        
        if model_name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feature_dim = 512
        elif model_name == 'resnet34':
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            feature_dim = 512
        elif model_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width] 单帧图像
        Returns:
            features: [batch, feature_dim] 特征向量
        """
        return self.features(x).squeeze(-1).squeeze(-1)


class VideoCNNLSTM(nn.Module):
    """
    CNN+LSTM 视频分类模型
    
    架构:
    1. CNN 提取每帧特征: [T, C, H, W] -> [T, feature_dim]
    2. LSTM 建模时序: [T, feature_dim] -> [hidden_dim]
    3. 全连接层分类: [hidden_dim] -> [num_classes]
    
    Args:
        num_classes: 分类数量 (疼痛等级)
        cnn_model: CNN 骨干网络名称
        cnn_pretrained: 是否使用预训练权重
        hidden_dim: LSTM 隐藏层维度
        num_layers: LSTM 层数
        dropout: Dropout 比率
        bidirectional: 是否使用双向 LSTM
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        cnn_model: str = 'resnet18',
        cnn_pretrained: bool = True,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        # CNN 特征提取器
        self.cnn = CNNFeatureExtractor(
            model_name=cnn_model,
            pretrained=cnn_pretrained
        )
        
        # LSTM 时序建模
        self.lstm = nn.LSTM(
            input_size=self.cnn.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # 分类头
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
    
    def forward(self, x):
        """
        Args:
            x: [batch, T, C, H, W] 视频帧序列
               batch: 批次大小
               T: 序列长度 (帧数)
               C: 通道数 (RGB=3)
               H, W: 图像高宽
        
        Returns:
            logits: [batch, num_classes] 分类输出
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # 1. CNN 提取每帧特征
        # 重塑: [batch*T, C, H, W]
        x = x.view(batch_size * seq_len, channels, height, width)
        frame_features = self.cnn(x)  # [batch*T, feature_dim]
        
        # 恢复序列结构: [batch, T, feature_dim]
        frame_features = frame_features.view(batch_size, seq_len, -1)
        
        # 2. LSTM 时序建模
        lstm_out, (hidden, cell) = self.lstm(frame_features)
        
        # 3. 使用最后一个时间步的输出进行分类
        if self.bidirectional:
            # 双向 LSTM: 拼接正向和反向的最后一个隐藏状态
            hidden_forward = hidden[-2]  # [batch, hidden_dim]
            hidden_backward = hidden[-1]  # [batch, hidden_dim]
            last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            last_hidden = hidden[-1]  # [batch, hidden_dim]
        
        # 4. 分类
        logits = self.classifier(last_hidden)
        
        return logits
    
    def get_num_params(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters())


def build_model(
    num_classes: int = 5,
    cnn_model: str = 'resnet18',
    hidden_dim: int = 256,
    num_lstm_layers: int = 2,
    dropout: float = 0.5,
    bidirectional: bool = False,
    pretrained: bool = True,
) -> VideoCNNLSTM:
    """
    构建 CNN+LSTM 模型的便捷函数
    
    Args:
        num_classes: 分类数
        cnn_model: CNN 骨干网络 ('resnet18', 'resnet34', 'resnet50')
        hidden_dim: LSTM 隐藏层维度
        num_lstm_layers: LSTM 层数
        dropout: Dropout 比率
        bidirectional: 是否双向 LSTM
        pretrained: 是否使用预训练 CNN 权重
    
    Returns:
        model: VideoCNNLSTM 实例
    """
    model = VideoCNNLSTM(
        num_classes=num_classes,
        cnn_model=cnn_model,
        cnn_pretrained=pretrained,
        hidden_dim=hidden_dim,
        num_layers=num_lstm_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    
    print(f"Model: CNN({cnn_model}) + LSTM(hidden={hidden_dim}, layers={num_lstm_layers})")
    print(f"Total parameters: {model.get_num_params():,}")
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = build_model(
        num_classes=5,
        cnn_model='resnet18',
        hidden_dim=256,
        num_lstm_layers=2,
    )
    
    # 模拟输入: batch=2, seq_len=16, RGB, 224x224
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    
    # 前向传播
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
