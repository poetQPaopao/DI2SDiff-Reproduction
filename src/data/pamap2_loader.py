"""
PAMAP2 Dataset Loader

This module provides functions to load and preprocess the PAMAP2 dataset
for human activity recognition tasks.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

def load_pamap2_data(
    data_dir: str = '/root/.jupyter/lab/workspaces/PAMAP2_Dataset/Protocol',
    window_size: int = 128,
    overlap: float = 0.5,
    normalize: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载PAMAP2数据集
    
    Args:
        data_dir: PAMAP2数据文件目录
        window_size: 滑动窗口大小
        overlap: 窗口重叠比例
        normalize: 是否标准化
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        X_train, X_test, y_train, y_test, train_subjects, test_subjects
    """
    
    # PAMAP2活动标签映射
    activity_map = {
        1: 0,  # lying
        2: 1,  # sitting
        3: 2,  # standing
        4: 3,  # walking
        5: 4,  # running
        6: 5,  # cycling
        7: 6,  # Nordic walking
        12: 7, # ascending stairs
        13: 7, # descending stairs (合并为climbing stairs)
        16: 3, # vacuum cleaning (归类为walking)
        17: 4, # ironing (归类为running)
        # 0, 8, 9, 10, 11, 14, 15, 18, 19, 20, 24 等其他活动将被过滤掉
    }
    
    # 传感器列定义 (基于PAMAP2数据格式，总共54列)
    # 列0: timestamp, 列1: activity_id, 列2: heart_rate
    # 列3-15: IMU hand, 列16-28: IMU chest, 列29-41: IMU ankle
    # 每个IMU包含: temp, 3x加速度, 3x陀螺仪, 3x磁力计, 4x四元数
    
    selected_columns = [
        # Chest IMU - 加速度计 (3轴) - 列17,18,19
        17, 18, 19,
        # Chest IMU - 陀螺仪 (3轴) - 列20,21,22
        20, 21, 22,
        # Chest IMU - 磁力计 (3轴) - 列23,24,25
        23, 24, 25,
        # Hand IMU - 加速度计 (3轴) - 列4,5,6
        4, 5, 6,
        # Ankle IMU - 加速度计 (3轴) - 列30,31,32
        30, 31, 32,
        # 心率 (列2)
        2,
    ]
    
    all_data = []
    all_labels = []
    all_subjects = []
    
    print("📊 开始加载PAMAP2数据...")
    
    # 遍历所有受试者文件
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.dat'):
            continue
            
        subject_id = int(filename.replace('subject', '').replace('.dat', ''))
        filepath = os.path.join(data_dir, filename)
        
        print(f"   处理受试者 {subject_id}: {filename}")
        
        try:
            # 读取数据文件
            data = pd.read_csv(filepath, sep=' ', header=None)
            print(f"    原始数据形状: {data.shape}")
            
            # 过滤有效活动标签
            valid_activities = data[1].isin(list(activity_map.keys()))
            data = data[valid_activities]
            
            if len(data) == 0:
                print(f"受试者 {subject_id} 无有效活动数据")
                continue
                
            # 提取传感器数据和标签
            sensor_data = data[selected_columns].values
            labels = data[1].values
            
            # 移除包含NaN的行
            valid_mask = ~np.isnan(sensor_data).any(axis=1)
            sensor_data = sensor_data[valid_mask]
            labels = labels[valid_mask]
            
            if len(sensor_data) == 0:
                print(f"     受试者 {subject_id} 处理后无有效数据")
                continue
                
            print(f"    处理后数据形状: {sensor_data.shape}")
            
            # 滑动窗口分割
            step_size = int(window_size * (1 - overlap))
            windows = []
            window_labels = []
            
            for i in range(0, len(sensor_data) - window_size + 1, step_size):
                window = sensor_data[i:i + window_size]
                window_label = labels[i:i + window_size]
                
                # 检查窗口内标签一致性 (至少80%相同)
                unique_labels, counts = np.unique(window_label, return_counts=True)
                if counts.max() / len(window_label) >= 0.8:
                    dominant_label = unique_labels[np.argmax(counts)]
                    if dominant_label in activity_map:
                        windows.append(window)
                        window_labels.append(activity_map[dominant_label])
            
            if len(windows) > 0:
                subject_windows = np.array(windows)
                subject_labels = np.array(window_labels)
                subject_ids = np.full(len(windows), subject_id)
                
                all_data.append(subject_windows)
                all_labels.append(subject_labels)
                all_subjects.append(subject_ids)
                
                print(f"     生成 {len(windows)} 个窗口")
            else:
                print(f"      受试者 {subject_id} 无有效窗口")
                
        except Exception as e:
            print(f"     处理受试者 {subject_id} 时出错: {e}")
            continue
    
    if not all_data:
        raise ValueError("❌ 未能加载任何有效数据")
    
    # 合并所有数据
    X = np.concatenate(all_data, axis=0)  # [N, T, C]
    y = np.concatenate(all_labels, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)
    
    print(f"\n📈 数据加载完成:")
    print(f"  总样本数: {X.shape[0]}")
    print(f"  数据形状: {X.shape}")
    print(f"  标签形状: {y.shape}")
    print(f"  受试者范围: {subjects.min()}-{subjects.max()}")
    
    # 转换为PyTorch格式 [N, C, T]
    X = X.transpose(0, 2, 1)
    
    # 标准化
    if normalize:
        print("🔄 执行数据标准化...")
        scaler = StandardScaler()
        N, C, T = X.shape
        X_reshaped = X.reshape(-1, C)
        X_normalized = scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(N, C, T)
    
    # 按样本随机划分训练/测试集（不按受试者划分）
    indices = np.arange(X.shape[0])
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    train_subjects = subjects[train_indices]
    test_subjects = subjects[test_indices]

    print(f"\n🎯 数据分割完成:")
    print(f"  训练集: {X_train.shape[0]} 样本, 受试者 {sorted(np.unique(train_subjects))}")
    print(f"  测试集: {X_test.shape[0]} 样本, 受试者 {sorted(np.unique(test_subjects))}")
    
    # 活动分布统计
    print(f"\n📊 活动分布统计:")
    activity_names = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'nordic_walking', 'stairs']
    for i, name in enumerate(activity_names):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        total_count = train_count + test_count
        if total_count > 0:
            print(f"  {name}: 训练 {train_count}, 测试 {test_count}, 总计 {total_count}")
    
    return X_train, X_test, y_train, y_test, train_subjects, test_subjects

def save_pamap2_npy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray, 
    y_test: np.ndarray,
    train_subjects: np.ndarray,
    test_subjects: np.ndarray,
    save_dir: str = 'pamap2_npy'
) -> None:
    """保存PAMAP2数据为npy格式"""
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(save_dir, 'subjects_train.npy'), train_subjects)
    np.save(os.path.join(save_dir, 'subjects_test.npy'), test_subjects)
    
    # 创建验证集 (使用测试集的副本)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_test)
    
    print(f"✅ 数据已保存到 {save_dir}/ 目录")

# 配置常量
PAMAP2_CONFIG = {
    'num_features': 16,
    'num_classes': 8,
    'window_size': 128,
    'overlap': 0.5,
    'activity_names': ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'nordic_walking', 'stairs'],
    'activity_map': {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 12: 7, 13: 7, 16: 3, 17: 4
    },
    'selected_columns': [17, 18, 19, 20, 21, 22, 23, 24, 25, 4, 5, 6, 30, 31, 32, 2]
}
