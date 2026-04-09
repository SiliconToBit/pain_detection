import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

# -----------------------------------------------------------------------------
# 心率单模态数据集类
# -----------------------------------------------------------------------------
class PainHRDataset(Dataset):
    def __init__(self, csv_path, split='train', seq_len=30):
        """
        参数:
            csv_path: 心率数据 CSV 文件路径
            split: 'train' 或 'val'（这里暂时不用，预留位置）
            seq_len: 心率序列的固定长度（比如 30 个点，对应 1 分钟）
        """
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.normalize_mode = os.getenv("NORMALIZE_MODE", "subject").lower()
        self.feature_mode = os.getenv("FEATURE_MODE", "basic").lower()
        self.has_subject_stats = {
            "person_hr_mean",
            "person_hr_std",
        }.issubset(set(self.df.columns))
        if self.feature_mode not in {"basic", "enhanced"}:
            raise ValueError(f"Unsupported FEATURE_MODE: {self.feature_mode}")

    @property
    def in_channels(self):
        return 1 if self.feature_mode == "basic" else 3

    def __len__(self):
        return len(self.df)

    def _parse_hr_sequence(self, hr_str):
        """解析 CSV 里的心率字符串，转成 numpy 数组"""
        # 按逗号分割，转成 float 列表
        hr_list = list(map(float, hr_str.split(',')))
        # 确保长度固定（不足的话用最后一个值填充，过长的话截断）
        if len(hr_list) < self.seq_len:
            hr_list += [hr_list[-1]] * (self.seq_len - len(hr_list))
        else:
            hr_list = hr_list[:self.seq_len]
        return np.array(hr_list, dtype=np.float32)

    def _normalize_hr(self, hr_seq):
        """心率归一化：Min-Max 归一化到 [0, 1]"""
        # 正常心率范围大概是 60-120，用这个范围做归一化更鲁棒
        min_hr = 60.0
        max_hr = 120.0
        hr_seq = (hr_seq - min_hr) / (max_hr - min_hr)
        # 限制在 [0, 1] 之间，避免异常值
        hr_seq = np.clip(hr_seq, 0.0, 1.0)
        return hr_seq

    def _subject_standardize(self, hr_seq, person_mean, person_std):
        """按受试者统计量做 z-score，再裁剪到稳定范围。"""
        std = max(float(person_std), 1e-6)
        z = (hr_seq - float(person_mean)) / std
        z = np.clip(z, -4.0, 4.0)
        # 映射到 [0, 1] 便于和现有模型输入尺度保持一致。
        return (z + 4.0) / 8.0

    @staticmethod
    def _tanh_unit_scale(values, scale):
        x = np.asarray(values, dtype=np.float32)
        return (np.tanh(x / scale) + 1.0) / 2.0

    def _build_enhanced_features(self, raw_hr_seq, base_seq):
        """构建增强通道: [基础序列, 一阶差分, 局部偏移]。"""
        # 一阶差分：体现瞬时变化趋势
        diff = np.diff(raw_hr_seq, prepend=raw_hr_seq[0])
        diff_feat = self._tanh_unit_scale(diff, scale=5.0)

        # 局部偏移：原序列减去短窗均值，体现局部波动
        kernel = np.ones(5, dtype=np.float32) / 5.0
        moving_avg = np.convolve(raw_hr_seq, kernel, mode="same")
        deviation = raw_hr_seq - moving_avg
        dev_feat = self._tanh_unit_scale(deviation, scale=5.0)

        return np.stack([base_seq, diff_feat, dev_feat], axis=0).astype(np.float32)

    @staticmethod
    def map_nrs_to_class(nrs_score):
        """将原始 NRS(1-8) 映射为 3 类标签(0/1/2)。"""
        score = int(float(nrs_score))
        if 1 <= score <= 3:
            return 0
        if 4 <= score <= 6:
            return 1
        if 7 <= score <= 8:
            return 2
        raise ValueError(f"Unexpected nrs_score: {nrs_score}")

    def __getitem__(self, idx):
        # 1. 获取心率序列字符串和 NRS 标签
        hr_str = self.df.iloc[idx]['hr_sequence']
        nrs_score = self.df.iloc[idx]['nrs_label']

        # 2. NRS 标签映射：1-3→0(轻度), 4-6→1(中度), 7-8→2(重度)
        label = self.map_nrs_to_class(nrs_score)

        # 3. 解析心率序列
        raw_hr_seq = self._parse_hr_sequence(hr_str)

        # 4. 归一化
        if self.normalize_mode == "subject" and self.has_subject_stats:
            person_mean = self.df.iloc[idx]["person_hr_mean"]
            person_std = self.df.iloc[idx]["person_hr_std"]
            base_seq = self._subject_standardize(raw_hr_seq, person_mean, person_std)
        else:
            base_seq = self._normalize_hr(raw_hr_seq)

        # 5. 组装输入通道
        if self.feature_mode == "enhanced":
            features = self._build_enhanced_features(raw_hr_seq, base_seq)
        else:
            features = np.expand_dims(base_seq, axis=0)

        # 1D-CNN 输入格式: [Batch, Channels, Length]
        hr_tensor = torch.tensor(features, dtype=torch.float32)

        return hr_tensor, torch.tensor(label, dtype=torch.long)