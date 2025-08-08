"""
Data preprocessing utilities for time-series HAR datasets.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union, List

def normalize_data(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = 'standard'
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    标准化时间序列数据
    
    Args:
        X_train: 训练数据 [N, C, T]
        X_test: 测试数据 [N, C, T] (可选)
        method: 标准化方法 ('standard' 或 'minmax')
    
    Returns:
        标准化后的数据
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    # 重塑数据进行标准化
    N, C, T = X_train.shape
    X_train_reshaped = X_train.reshape(-1, C)
    X_train_normalized = scaler.fit_transform(X_train_reshaped)
    X_train_normalized = X_train_normalized.reshape(N, C, T)
    
    if X_test is not None:
        N_test, C, T = X_test.shape
        X_test_reshaped = X_test.reshape(-1, C)
        X_test_normalized = scaler.transform(X_test_reshaped)
        X_test_normalized = X_test_normalized.reshape(N_test, C, T)
        return X_train_normalized, X_test_normalized
    
    return X_train_normalized

def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 128,
    overlap: float = 0.5,
    min_label_consistency: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用滑动窗口分割时间序列数据
    
    Args:
        data: 时间序列数据 [T, C]
        labels: 对应的标签 [T]
        window_size: 窗口大小
        overlap: 重叠比例
        min_label_consistency: 窗口内标签一致性最小要求
    
    Returns:
        windows: [N, C, T], window_labels: [N]
    """
    step_size = int(window_size * (1 - overlap))
    windows = []
    window_labels = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]  # [T, C]
        window_label = labels[i:i + window_size]
        
        # 检查标签一致性
        unique_labels, counts = np.unique(window_label, return_counts=True)
        if counts.max() / len(window_label) >= min_label_consistency:
            dominant_label = unique_labels[np.argmax(counts)]
            windows.append(window.T)  # 转换为 [C, T]
            window_labels.append(dominant_label)
    
    if len(windows) == 0:
        return np.empty((0, data.shape[1], window_size)), np.empty(0)
    
    return np.array(windows), np.array(window_labels)

def train_test_split_subjects(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按受试者划分训练/测试集
    
    Args:
        X: 特征数据 [N, C, T]
        y: 标签 [N]
        subjects: 受试者ID [N]
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        X_train, X_test, y_train, y_test, train_subjects, test_subjects
    """
    unique_subjects = np.unique(subjects)
    train_subjects_ids, test_subjects_ids = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )
    
    train_mask = np.isin(subjects, train_subjects_ids)
    test_mask = np.isin(subjects, test_subjects_ids)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    train_subjects = subjects[train_mask]
    test_subjects = subjects[test_mask]
    
    return X_train, X_test, y_train, y_test, train_subjects, test_subjects

def augment_time_series(
    X: np.ndarray,
    augmentation_factor: int = 2,
    noise_level: float = 0.01,
    jitter_sigma: float = 0.03,
    scaling_sigma: float = 0.1
) -> np.ndarray:
    """
    时间序列数据增强
    
    Args:
        X: 输入数据 [N, C, T]
        augmentation_factor: 增强倍数
        noise_level: 噪声水平
        jitter_sigma: 抖动标准差
        scaling_sigma: 缩放标准差
    
    Returns:
        增强后的数据
    """
    N, C, T = X.shape
    augmented_data = []
    
    for _ in range(augmentation_factor):
        # 添加噪声
        noise = np.random.normal(0, noise_level, X.shape)
        X_noise = X + noise
        
        # 时间抖动
        jitter = np.random.normal(0, jitter_sigma, (N, C, 1))
        X_jitter = X + jitter
        
        # 幅度缩放
        scaling = np.random.normal(1, scaling_sigma, (N, C, 1))
        X_scaled = X * scaling
        
        augmented_data.extend([X_noise, X_jitter, X_scaled])
    
    return np.concatenate([X] + augmented_data, axis=0)

def filter_by_activity(
    X: np.ndarray,
    y: np.ndarray,
    activity_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按活动类别过滤数据
    
    Args:
        X: 特征数据 [N, C, T]
        y: 标签 [N]
        activity_ids: 要保留的活动ID列表
    
    Returns:
        过滤后的数据和标签
    """
    mask = np.isin(y, activity_ids)
    return X[mask], y[mask]

def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = 'undersample'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    类别平衡处理
    
    Args:
        X: 特征数据 [N, C, T]
        y: 标签 [N]
        strategy: 'undersample' 或 'oversample'
    
    Returns:
        平衡后的数据和标签
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if strategy == 'undersample':
        min_count = counts.min()
        balanced_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            selected = np.random.choice(cls_indices, min_count, replace=False)
            balanced_indices.extend(selected)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        return X[balanced_indices], y[balanced_indices]
    
    elif strategy == 'oversample':
        max_count = counts.max()
        balanced_X = []
        balanced_y = []
        
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            cls_X = X[cls_indices]
            
            if len(cls_indices) < max_count:
                # 重复采样
                oversample_indices = np.random.choice(
                    len(cls_indices), max_count, replace=True
                )
                cls_X = cls_X[oversample_indices]
            
            balanced_X.append(cls_X)
            balanced_y.extend([cls] * max_count)
        
        return np.concatenate(balanced_X, axis=0), np.array(balanced_y)
    
    else:
        raise ValueError(f"不支持的策略: {strategy}")

def calculate_dataset_stats(X: np.ndarray, y: np.ndarray) -> dict:
    """
    计算数据集统计信息
    
    Args:
        X: 特征数据 [N, C, T]
        y: 标签 [N]
    
    Returns:
        统计信息字典
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    
    stats = {
        'num_samples': X.shape[0],
        'num_features': X.shape[1],
        'sequence_length': X.shape[2],
        'num_classes': len(unique_classes),
        'class_distribution': dict(zip(unique_classes, counts)),
        'feature_means': X.mean(axis=(0, 2)),
        'feature_stds': X.std(axis=(0, 2)),
        'data_range': {
            'min': X.min(),
            'max': X.max(),
            'mean': X.mean(),
            'std': X.std()
        }
    }
    
    return stats
