import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from insightface.app import FaceAnalysis

# -----------------------------------------------------------------------------
# 1. RetinaFace 人脸检测器（全局单例，避免重复加载）
# -----------------------------------------------------------------------------
_retinaface_detector = None

def get_retinaface_detector():
    """
    获取 RetinaFace 检测器单例
    使用 insightface 库，内部使用 RetinaFace 模型
    """
    global _retinaface_detector
    if _retinaface_detector is None:
        # 初始化人脸分析器，使用 retinaface 检测
        _retinaface_detector = FaceAnalysis(
            name='buffalo_l',  # 包含 RetinaFace 检测器
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _retinaface_detector.prepare(ctx_id=0, det_size=(640, 640))
    return _retinaface_detector

def retinaface_crop(frame, expand_ratio=0.2):
    """
    使用 RetinaFace 精准检测人脸并裁剪
    
    参数:
        frame: BGR 格式的图像帧
        expand_ratio: 人脸框扩展比例，避免裁剪过紧
    
    返回:
        裁剪后的人脸图像，如果检测失败则返回中心裁剪
    """
    h, w = frame.shape[:2]
    
    try:
        detector = get_retinaface_detector()
        faces = detector.get(frame)
        
        if len(faces) > 0:
            # 取置信度最高的人脸
            face = max(faces, key=lambda x: x.det_score)
            bbox = face.bbox.astype(int)
            
            # 提取边界框坐标
            x1, y1, x2, y2 = bbox
            
            # 计算扩展后的边界框
            face_w = x2 - x1
            face_h = y2 - y1
            expand_w = int(face_w * expand_ratio)
            expand_h = int(face_h * expand_ratio)
            
            # 扩展边界框并限制在图像范围内
            x1 = max(0, x1 - expand_w)
            y1 = max(0, y1 - expand_h)
            x2 = min(w, x2 + expand_w)
            y2 = min(h, y2 + expand_h)
            
            # 裁剪人脸区域
            face_crop = frame[y1:y2, x1:x2]
            return face_crop
    except Exception as e:
        print(f"RetinaFace 检测失败: {e}，使用中心裁剪")
    
    # 检测失败时的回退策略：中心裁剪
    crop_h, crop_w = int(h * 0.7), int(w * 0.7)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    return frame[start_h:start_h+crop_h, start_w:start_w+crop_w]

# -----------------------------------------------------------------------------
# 2. 视频数据集类
# -----------------------------------------------------------------------------
class PainVideoDataset(Dataset):
    def __init__(self, csv_path, split='train', num_frames=16, img_size=224):
        """
        参数:
            csv_path: 标签文件路径，包含 video_path 和 nrs_label 两列
            split: 'train' 或 'val'，用于区分数据增强
            num_frames: 每段视频抽取的帧数
            img_size: 输入模型的图像尺寸
        """
        self.df = pd.read_csv(csv_path)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size

        # 训练集数据增强（提升泛化能力）
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度调整
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 归一化
            ])
        # 验证集：仅做基础预处理
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def _load_video(self, video_path):
        """读取视频，均匀抽帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 计算抽帧间隔
        interval = max(1, total_frames // self.num_frames)
        
        frames = []
        for i in range(self.num_frames):
            # 定位到目标帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if not ret:
                # 如果读取失败，用最后一帧填充
                if frames:
                    frame = frames[-1]
                else:
                    frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
            # 人脸裁剪（使用 RetinaFace 精准检测）
            frame = retinaface_crop(frame)
            # BGR 转 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames

    def __getitem__(self, idx):
        # 获取视频路径和标签
        video_path = self.df.iloc[idx]['video_path']
        # NRS 标签映射：0→0, 1-3→1, 4-6→2, 7-10→3
        nrs_score = self.df.iloc[idx]['nrs_label']
        if nrs_score == 0:
            label = 0
        elif 1 <= nrs_score <= 3:
            label = 1
        elif 4 <= nrs_score <= 6:
            label = 2
        else:
            label = 3

        # 读取视频帧
        frames = self._load_video(video_path)
        
        # 对每一帧做预处理
        processed_frames = []
        for frame in frames:
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
        
        # 堆叠成张量：[T, C, H, W]
        video_tensor = torch.stack(processed_frames)
        # 转换为 Video Swin 输入格式：[C, T, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)