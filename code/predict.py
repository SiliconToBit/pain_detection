"""
视频疼痛识别预测脚本
- 加载训练好的 CNN+LSTM 模型
- 对视频帧序列进行疼痛等级预测
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional

from model_video import build_model
from dataset_video import get_val_transform


class VideoPainPredictor:
    """
    视频疼痛等级预测器
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 运行设备 ('cuda' 或 'cpu')
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # 加载检查点
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取配置
        self.config = checkpoint['config']
        self.num_classes = self.config['num_classes']
        
        # 构建模型
        self.model = build_model(
            num_classes=self.config['num_classes'],
            cnn_model=self.config['cnn_model'],
            hidden_dim=self.config['hidden_dim'],
            num_lstm_layers=self.config['num_lstm_layers'],
            dropout=0.0,  # 预测时不使用 dropout
            bidirectional=self.config['bidirectional'],
            pretrained=False,
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 创建变换
        self.transform = get_val_transform(
            modality=self.config['modality'],
            img_size=self.config['img_size'],
        )
        
        self.seq_len = self.config['seq_len']
        self.modality = self.config['modality']
        
        print(f"模型加载完成: {checkpoint['best_kappa']:.4f} kappa")
    
    def _load_frame(self, frame_path: str) -> Image.Image:
        """加载单帧图像"""
        img = Image.open(frame_path)
        
        if self.modality == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            if img.mode != 'L':
                img = img.convert('L')
        
        return img
    
    def _sample_frames(self, frame_paths: List[str]) -> List[str]:
        """采样帧序列"""
        num_frames = len(frame_paths)
        
        if num_frames <= self.seq_len:
            # 帧数不足,重复最后一帧
            sampled = frame_paths.copy()
            while len(sampled) < self.seq_len:
                sampled.append(frame_paths[-1])
            return sampled
        
        # 均匀采样
        indices = np.linspace(0, num_frames - 1, self.seq_len, dtype=int)
        return [frame_paths[i] for i in indices]
    
    def predict_frames(self, frame_paths: List[str]) -> dict:
        """
        从帧路径列表预测疼痛等级
        
        Args:
            frame_paths: 图像文件路径列表
        
        Returns:
            result: {
                'predicted_class': int,
                'probabilities': np.array,
                'confidence': float,
            }
        """
        # 采样帧
        sampled_paths = self._sample_frames(frame_paths)
        
        # 加载并变换帧
        frames = []
        for path in sampled_paths:
            img = self._load_frame(path)
            frame_tensor = self.transform(img)
            frames.append(frame_tensor)
        
        # 堆叠成序列: [1, T, C, H, W]
        frames = torch.stack(frames, dim=0).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(frames)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
        
        return {
            'predicted_class': predicted_class,
            'probabilities': probabilities,
            'confidence': confidence,
        }
    
    def predict_video(self, video_dir: str) -> dict:
        """
        从视频目录预测疼痛等级
        
        Args:
            video_dir: 包含帧图像的目录路径
        
        Returns:
            result: 预测结果字典
        """
        video_dir = Path(video_dir)
        
        # 根据模态查找图像文件
        if self.modality == 'rgb':
            frame_paths = sorted(list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png')))
        elif self.modality == 'depth':
            frame_paths = sorted(list(video_dir.glob('D/*.png')) + list(video_dir.glob('D/*.jpg')))
        elif self.modality == 'thermal':
            frame_paths = sorted(list(video_dir.glob('T/*.png')) + list(video_dir.glob('T/*.jpg')))
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
        
        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in {video_dir}")
        
        return self.predict_frames([str(p) for p in frame_paths])


def predict_single_video(
    checkpoint_path: str,
    video_dir: str,
    device: str = None,
) -> dict:
    """
    便捷函数: 预测单个视频
    
    Args:
        checkpoint_path: 模型检查点路径
        video_dir: 视频帧目录
        device: 设备
    
    Returns:
        result: 预测结果
    """
    predictor = VideoPainPredictor(checkpoint_path, device)
    return predictor.predict_video(video_dir)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='视频疼痛等级预测')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频帧目录')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 预测
    result = predict_single_video(
        checkpoint_path=args.checkpoint,
        video_dir=args.video_dir,
        device=args.device,
    )
    
    # 输出结果
    print(f"\n预测结果:")
    print(f"  疼痛等级: {result['predicted_class']}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  概率分布: {result['probabilities']}")
