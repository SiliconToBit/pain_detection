"""
视频帧序列数据集
- 从 txt 文件读取数据列表
- 组织成帧序列
- 支持多模态 (RGB/Depth/Thermal)
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable


class VideoFrameSequenceDataset(Dataset):
    """
    视频帧序列数据集
    
    从 txt 文件读取数据列表，组织成帧序列用于训练
    
    txt 文件格式: <file_path> <label> <video_id>
    
    Args:
        txt_path: 数据列表文件路径 (如 train_rgb.txt)
        seq_len: 序列长度 (每个视频采样多少帧)
        num_classes: 分类数
        transform: 图像变换
        modality: 输入模态 ('rgb', 'depth', 'thermal')
        sample_mode: 帧采样模式 ('uniform', 'random', 'all')
        min_frames: 最少帧数 (少于这个数量的视频会被跳过)
        max_frames: 最大帧数 (超过这个数量会被截断)
        return_video_id: 是否返回 video_id
    """
    
    def __init__(
        self,
        txt_path: str,
        seq_len: int = 16,
        num_classes: int = 5,
        transform: Optional[Callable] = None,
        modality: str = 'rgb',
        sample_mode: str = 'uniform',
        min_frames: int = 5,
        max_frames: int = 100,
        return_video_id: bool = False,
    ):
        self.txt_path = txt_path
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.transform = transform
        self.modality = modality
        self.sample_mode = sample_mode
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.return_video_id = return_video_id
        
        # 解析 txt 文件，按 video_id 组织数据
        self.video_data = self._parse_txt_file(txt_path)
        self.video_ids = list(self.video_data.keys())
        
        print(f"Loaded {len(self.video_ids)} videos from {txt_path}")
        print(f"Modality: {modality}, Sequence length: {seq_len}")
        
        # 统计标签分布
        self._print_label_distribution()
    
    def _parse_txt_file(self, txt_path: str) -> Dict[int, Dict]:
        """
        解析 txt 文件，按 video_id 组织帧数据
        
        Returns:
            video_data: {video_id: {'frames': [paths], 'label': int}}
        """
        video_data = {}
        
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                file_path = parts[0]
                label = int(parts[1])
                video_id = int(parts[2])
                
                if video_id not in video_data:
                    video_data[video_id] = {
                        'frames': [],
                        'label': label,
                    }
                
                video_data[video_id]['frames'].append(file_path)
        
        # 对每个视频的帧进行排序
        for vid in video_data:
            video_data[vid]['frames'].sort()
        
        # 过滤帧数不足的视频
        filtered_data = {}
        for vid, data in video_data.items():
            num_frames = len(data['frames'])
            if num_frames >= self.min_frames:
                # 截断过长的视频
                if num_frames > self.max_frames:
                    data['frames'] = data['frames'][:self.max_frames]
                filtered_data[vid] = data
        
        return filtered_data
    
    def _print_label_distribution(self):
        """打印标签分布"""
        label_counts = {}
        for vid, data in self.video_data.items():
            label = data['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Label distribution: {label_counts}")
    
    def _sample_frames(self, frames: List[str]) -> List[str]:
        """
        从帧列表中采样指定数量的帧
        
        Args:
            frames: 所有帧路径列表
        Returns:
            sampled_frames: 采样后的帧路径列表
        """
        num_frames = len(frames)
        
        if num_frames <= self.seq_len:
            # 帧数不足，重复最后一帧
            sampled = frames.copy()
            while len(sampled) < self.seq_len:
                sampled.append(frames[-1])
            return sampled
        
        if self.sample_mode == 'uniform':
            # 均匀采样
            indices = np.linspace(0, num_frames - 1, self.seq_len, dtype=int)
            return [frames[i] for i in indices]
        
        elif self.sample_mode == 'random':
            # 随机采样
            indices = random.sample(range(num_frames), self.seq_len)
            indices.sort()
            return [frames[i] for i in indices]
        
        else:
            # 默认取前 seq_len 帧
            return frames[:self.seq_len]
    
    def _load_frame(self, frame_path: str) -> np.ndarray:
        """
        加载单帧图像
        
        Args:
            frame_path: 图像路径
        Returns:
            image: numpy array [H, W, C]
        """
        try:
            img = Image.open(frame_path)
            
            # 根据模态处理图像
            if self.modality == 'rgb':
                # RGB 保持 3 通道
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            elif self.modality == 'depth':
                # Depth 通常是单通道
                if img.mode != 'L':
                    img = img.convert('L')
            elif self.modality == 'thermal':
                # Thermal 通常是单通道
                if img.mode != 'L':
                    img = img.convert('L')
            
            return np.array(img)
        
        except Exception as e:
            print(f"Error loading frame {frame_path}: {e}")
            # 返回空白图像
            if self.modality == 'rgb':
                return np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                return np.zeros((224, 224), dtype=np.uint8)
    
    def __len__(self) -> int:
        return len(self.video_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本
        
        Returns:
            frames: [T, C, H, W] 帧序列
            label: 疼痛等级
        """
        video_id = self.video_ids[idx]
        data = self.video_data[video_id]
        
        # 采样帧
        sampled_frames = self._sample_frames(data['frames'])
        
        # 加载帧
        frame_arrays = []
        for frame_path in sampled_frames:
            frame = self._load_frame(frame_path)
            frame_arrays.append(frame)
        
        # 堆叠成序列
        frames = np.stack(frame_arrays, axis=0)  # [T, H, W, C] 或 [T, H, W]
        
        # 应用变换
        if self.transform:
            transformed_frames = []
            for i in range(frames.shape[0]):
                if self.modality == 'rgb':
                    # RGB: [H, W, C]
                    transformed = self.transform(Image.fromarray(frames[i]))
                else:
                    # Depth/Thermal: [H, W] -> 转为 PIL Image
                    transformed = self.transform(Image.fromarray(frames[i]))
                transformed_frames.append(transformed)
            frames = torch.stack(transformed_frames, dim=0)  # [T, C, H, W]
        else:
            # 默认处理: 转为 tensor
            if self.modality == 'rgb':
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
            else:
                frames = torch.from_numpy(frames).unsqueeze(1).float() / 255.0  # [T, 1, H, W]
                # 复制为 3 通道以兼容 ResNet
                frames = frames.repeat(1, 3, 1, 1)
        
        label = data['label']
        
        if self.return_video_id:
            return frames, label, video_id
        
        return frames, label


class MultiModalVideoDataset(Dataset):
    """
    多模态视频帧序列数据集
    
    同时加载 RGB、Depth、Thermal 数据
    
    Args:
        txt_paths: 各模态的 txt 文件路径字典 {'rgb': path, 'depth': path, 'thermal': path}
        seq_len: 序列长度
        num_classes: 分类数
        transform: 图像变换
        modalities: 要使用的模态列表
        sample_mode: 帧采样模式
    """
    
    def __init__(
        self,
        txt_paths: Dict[str, str],
        seq_len: int = 16,
        num_classes: int = 5,
        transform: Optional[Callable] = None,
        modalities: List[str] = ['rgb'],
        sample_mode: str = 'uniform',
    ):
        self.txt_paths = txt_paths
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.transform = transform
        self.modalities = modalities
        self.sample_mode = sample_mode
        
        # 加载各模态数据
        self.datasets = {}
        for mod in modalities:
            if mod in txt_paths:
                self.datasets[mod] = VideoFrameSequenceDataset(
                    txt_path=txt_paths[mod],
                    seq_len=seq_len,
                    num_classes=num_classes,
                    transform=transform,
                    modality=mod,
                    sample_mode=sample_mode,
                )
        
        # 确保各模态的 video_id 一致
        self._align_video_ids()
        
        print(f"MultiModalVideoDataset: {len(self.video_ids)} videos, modalities: {modalities}")
    
    def _align_video_ids(self):
        """对齐各模态的 video_id"""
        # 取第一个模态的 video_id 作为基准
        first_mod = self.modalities[0]
        self.video_ids = self.datasets[first_mod].video_ids
        
        # 过滤掉其他模态中缺失的 video_id
        for mod in self.modalities[1:]:
            mod_ids = set(self.datasets[mod].video_ids)
            self.video_ids = [vid for vid in self.video_ids if vid in mod_ids]
        
        # 更新各数据集的 video_ids
        for mod in self.modalities:
            self.datasets[mod].video_ids = self.video_ids
    
    def __len__(self) -> int:
        return len(self.video_ids)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        获取单个样本
        
        Returns:
            frames_dict: {'rgb': [T, C, H, W], 'depth': [T, C, H, W], ...}
            label: 疼痛等级
        """
        video_id = self.video_ids[idx]
        
        frames_dict = {}
        label = None
        
        for mod in self.modalities:
            # 找到该 video_id 在各数据集中的索引
            mod_idx = self.datasets[mod].video_ids.index(video_id)
            frames, mod_label = self.datasets[mod][mod_idx]
            frames_dict[mod] = frames
            
            if label is None:
                label = mod_label
        
        return frames_dict, label


def get_default_transform(modality: str = 'rgb', img_size: int = 224):
    """获取默认图像变换"""
    from torchvision import transforms
    
    if modality == 'rgb':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Depth/Thermal: 单通道
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # 复制为 3 通道
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])


def get_val_transform(modality: str = 'rgb', img_size: int = 224):
    """获取验证/测试图像变换"""
    from torchvision import transforms
    
    if modality == 'rgb':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])


if __name__ == '__main__':
    # 测试数据集
    from torchvision import transforms
    
    txt_path = '/home/gm/Workspace/ai-projects/pain_detection/data/mintpain/train_rgb.txt'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = VideoFrameSequenceDataset(
        txt_path=txt_path,
        seq_len=16,
        num_classes=5,
        transform=transform,
        modality='rgb',
        sample_mode='uniform',
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试加载
    frames, label = dataset[0]
    print(f"Frames shape: {frames.shape}, Label: {label}")