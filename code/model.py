import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 心率单模态模型：支持 legacy(1D-CNN+BiLSTM) 与 causal(因果CNN+GRU)
# -----------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    """Left-padded causal conv: output[t] depends on <= t only."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class HRSingleModalModel(nn.Module):
    def __init__(
        self,
        seq_len=30,
        num_classes=4,
        in_channels=1,
        output_mode="ce",
        model_arch="causal_gru",
        hidden_dim=64,
    ):
        """
        参数:
            seq_len: 心率序列长度（比如 30）
            num_classes: NRS 4 分类
            in_channels: 输入通道数（基础模式1通道，增强模式多通道）
            output_mode: "ce"(普通分类) 或 "coral"(有序回归)
            model_arch: "causal_gru"(默认) 或 "legacy_bilstm"
        """
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.output_mode = output_mode.lower()
        self.model_arch = model_arch.lower()
        if self.output_mode not in {"ce", "coral"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}")
        if self.model_arch not in {"causal_gru", "legacy_bilstm"}:
            raise ValueError(f"Unsupported model_arch: {model_arch}")

        out_dim = num_classes if self.output_mode == "ce" else (num_classes - 1)

        if self.model_arch == "legacy_bilstm":
            # 保留旧结构，兼容历史 checkpoint。
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            self.temporal = nn.LSTM(
                input_size=64,
                hidden_size=hidden_dim,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
            temporal_out_dim = hidden_dim * 2
        else:
            # 实时架构：严格因果卷积 + 单向 GRU + GroupNorm（替代 BatchNorm）
            self.conv_layers = nn.Sequential(
                CausalConv1d(in_channels=in_channels, out_channels=32, kernel_size=5),
                nn.GroupNorm(num_groups=8, num_channels=32),
                nn.ReLU(),
                CausalConv1d(in_channels=32, out_channels=64, kernel_size=3),
                nn.GroupNorm(num_groups=8, num_channels=64),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.temporal = nn.GRU(
                input_size=64,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=False,
                batch_first=True,
            )
            temporal_out_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def _encode(self, x, h_0=None):
        conv_feat = self.conv_layers(x)
        temporal_in = conv_feat.permute(0, 2, 1)

        if self.model_arch == "legacy_bilstm":
            temporal_out, _ = self.temporal(temporal_in)
            # 正向最后一步 + 反向第一步
            hidden_dim = temporal_out.size(-1) // 2
            last_feat = torch.cat(
                [temporal_out[:, -1, :hidden_dim], temporal_out[:, 0, hidden_dim:]],
                dim=-1,
            )
            return last_feat, None

        temporal_out, h_n = self.temporal(temporal_in, h_0)
        last_feat = temporal_out[:, -1, :]
        return last_feat, h_n

    def forward(self, x):
        """
        前向传播
        参数 x: [Batch_Size, C, Seq_Len]，例如 [8, 1, 30] 或 [8, 3, 30]
        """
        feat, _ = self._encode(x)
        return self.classifier(feat)

    def forward_with_state(self, x, h_0=None):
        """用于实时推理：返回 logits、GRU 隐状态、最后时刻特征。"""
        feat, h_n = self._encode(x, h_0=h_0)
        logits = self.classifier(feat)
        return logits, h_n, feat