import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Video Swin Transformer 产痛评估模型
# -----------------------------------------------------------------------------
class VideoSwinPainModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, freeze_backbone=False):
        """
        参数:
            num_classes: NRS 4 分类
            pretrained: 是否加载预训练权重
            freeze_backbone: 是否冻结骨干网络（小数据集建议冻结）
        """
        super().__init__()
        
        # 1. 加载预训练的 Video Swin Tiny
        # 模型来源：Facebook Research PyTorchVideo
        self.video_backbone = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model="video_swin_tiny",
            pretrained=pretrained
        )
        
        # 2. 冻结骨干网络（可选）
        if freeze_backbone:
            for name, param in self.video_backbone.named_parameters():
                # 只冻结前 50% 的层，保留最后几层微调
                if "blocks.0" in name or "blocks.1" in name or "patch_embed" in name:
                    param.requires_grad = False
        
        # 3. 移除原模型的分类头（原模型是 Kinetics 400 分类）
        # Video Swin Tiny 的输出特征维度是 768
        self.video_backbone.head = nn.Identity()
        
        # 4. 定义我们的 NRS 4 分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),  # Dropout 防止过拟合
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        参数 x: [Batch_Size, Channels, Num_Frames, Height, Width]
               例如: [8, 3, 16, 224, 224]
        """
        # 通过 Video Swin 提取特征
        feat = self.video_backbone(x)  # [Batch_Size, 768]
        # 通过分类头
        out = self.classifier(feat)    # [Batch_Size, 4]
        return out