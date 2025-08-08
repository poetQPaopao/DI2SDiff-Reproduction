"""
TS-TCC Model Implementation

Time-Series representation learning via Temporal and Contextual Contrasting
Based on the original TS-TCC paper and implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 8, stride: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TSTCCEncoder(nn.Module):
    """TS-TCC编码器"""
    def __init__(self, input_channels: int = 12, mid_channels: int = 64, final_out_channels: int = 128):
        super(TSTCCEncoder, self).__init__()
        
        self.conv_block1 = ConvBlock(input_channels, mid_channels)
        self.conv_block2 = ConvBlock(mid_channels, mid_channels)
        self.conv_block3 = ConvBlock(mid_channels, final_out_channels)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # Global Average Pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return x

class TSTCCModel(nn.Module):
    """完整的TS-TCC模型"""
    def __init__(
        self, 
        input_channels: int = 12, 
        num_classes: int = 6,
        mid_channels: int = 64,
        final_out_channels: int = 128,
        dropout: float = 0.35
    ):
        super(TSTCCModel, self).__init__()
        
        self.encoder = TSTCCEncoder(input_channels, mid_channels, final_out_channels)
        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(final_out_channels, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, sequence_length]
        Returns:
            predictions, features
        """
        features = self.encoder(x)
        features_dropout = self.dropout(features)
        predictions = self.logits(features_dropout)
        return predictions, features

class TemporalContrastiveModel(nn.Module):
    """时间对比学习模型"""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 100, output_dim: int = 50):
        super(TemporalContrastiveModel, self).__init__()
        
        self.context_vector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, features):
        """
        Args:
            features: [batch_size, feature_dim]
        Returns:
            context vectors
        """
        return self.context_vector(features)
    
    def context(self, features):
        """获取上下文向量"""
        return self.forward(features)

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    def __init__(self, temperature: float = 0.2, use_cosine_similarity: bool = True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        
    def forward(self, zis: torch.Tensor, zjs: torch.Tensor):
        """
        计算NT-Xent损失
        Args:
            zis: [batch_size, feature_dim] - 第一个增强视图的特征
            zjs: [batch_size, feature_dim] - 第二个增强视图的特征
        """
        batch_size = zis.shape[0]
        
        # 标准化特征
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        
        # 计算相似度矩阵
        representations = torch.cat([zis, zjs], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )
        
        # 创建标签 - 正样本对的索引
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        
        # 移除自相似度
        mask = torch.eye(2 * batch_size, dtype=bool, device=similarity_matrix.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        
        # 计算损失
        negatives = similarity_matrix
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=zis.device)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        
        return loss

class DataAugmentation:
    """数据增强工具类"""
    
    @staticmethod
    def jitter(x: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
        """添加高斯噪声"""
        return x + torch.normal(mean=0, std=sigma, size=x.shape, device=x.device)
    
    @staticmethod
    def scaling(x: torch.Tensor, sigma: float = 1.1) -> torch.Tensor:
        """随机缩放"""
        factor = torch.normal(mean=1.0, std=sigma, size=(x.shape[0], x.shape[1], 1), device=x.device)
        return x * factor
    
    @staticmethod
    def rotation(x: torch.Tensor) -> torch.Tensor:
        """随机旋转（翻转）"""
        flip = torch.randint(0, 2, (x.shape[0], x.shape[1], 1), device=x.device) * 2 - 1
        return x * flip
    
    @staticmethod
    def permutation(x: torch.Tensor, max_segments: int = 5, seg_mode: str = "random") -> torch.Tensor:
        """时间序列排列"""
        orig_steps = x.shape[2]
        
        if seg_mode == "random":
            # 随机分段数
            num_segs = torch.randint(1, max_segments + 1, (1,)).item()
        else:
            num_segs = max_segments
            
        if num_segs > 1:
            # 计算分段点
            seg_size = orig_steps // num_segs
            segments = []
            
            for i in range(num_segs):
                start_idx = i * seg_size
                end_idx = start_idx + seg_size if i < num_segs - 1 else orig_steps
                segments.append(x[:, :, start_idx:end_idx])
            
            # 随机排列分段
            perm_segments = torch.randperm(num_segs)
            permuted_x = torch.cat([segments[i] for i in perm_segments], dim=2)
            return permuted_x
        
        return x
    
    @staticmethod
    def magnitude_warp(x: torch.Tensor, sigma: float = 0.2, knot: int = 4) -> torch.Tensor:
        """幅度扭曲"""
        orig_steps = x.shape[2]
        
        # 生成扭曲曲线
        random_warps = torch.normal(mean=1.0, std=sigma, size=(x.shape[0], knot + 2), device=x.device)
        warp_steps = torch.linspace(0, orig_steps - 1, knot + 2, device=x.device)
        
        # 插值到原始长度
        ret = torch.zeros_like(x)
        for i in range(x.shape[0]):
            warper = torch.interp(
                torch.arange(orig_steps, dtype=torch.float, device=x.device),
                warp_steps, 
                random_warps[i]
            )
            ret[i] = x[i] * warper.unsqueeze(0)
        
        return ret

def apply_augmentations(x: torch.Tensor, augmentation_list: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用数据增强
    
    Args:
        x: 输入张量 [batch_size, channels, sequence_length]
        augmentation_list: 增强方法列表
        
    Returns:
        两个增强后的视图
    """
    aug1 = x.clone()
    aug2 = x.clone()
    
    # 随机选择增强方法
    aug_methods = {
        'jitter': DataAugmentation.jitter,
        'scaling': DataAugmentation.scaling, 
        'rotation': DataAugmentation.rotation,
        'permutation': DataAugmentation.permutation,
        'magnitude_warp': DataAugmentation.magnitude_warp
    }
    
    # 为每个视图应用不同的增强
    for aug_name in augmentation_list:
        if aug_name in aug_methods:
            if torch.rand(1) > 0.5:  # 50%概率应用增强
                aug1 = aug_methods[aug_name](aug1)
            if torch.rand(1) > 0.5:
                aug2 = aug_methods[aug_name](aug2)
    
    return aug1, aug2

# 模型配置
TSTCC_CONFIG = {
    'motionsense': {
        'input_channels': 12,
        'num_classes': 6,
        'mid_channels': 64,
        'final_out_channels': 128,
        'dropout': 0.35
    },
    'pamap2': {
        'input_channels': 16,
        'num_classes': 8,
        'mid_channels': 64,
        'final_out_channels': 128,
        'dropout': 0.35
    }
}
