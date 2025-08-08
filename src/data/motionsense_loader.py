"""
MotionSense Dataset Loader

This module provides functions to load and preprocess the MotionSense dataset
with sliding window segmentation for time-series analysis.
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional

def get_ds_infos_windowed(data_dir: str) -> np.ndarray:
    """
    获取受试者信息
    
    Args:
        data_dir: MotionSense数据目录路径
        
    Returns:
        受试者信息数组 [subject_id, weight, height, age, gender]
    """
    # 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
    dss = np.genfromtxt(os.path.join(data_dir, "data_subjects_info.csv"), delimiter=',')
    dss = dss[1:]  # 跳过header
    print("----> Data subjects information is imported.")
    return dss

def create_windowed_time_series(
    data_dir: str,
    num_features: int = 12,
    num_act_labels: int = 6, 
    num_gen_labels: int = 1,
    label_codes: Optional[Dict[str, int]] = None,
    trial_codes: Optional[Dict[str, List[int]]] = None,
    window_size: int = 128,
    overlap: float = 0.5,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建基于滑动窗口的MotionSense时间序列数据
    
    Args:
        data_dir: MotionSense数据目录路径
        num_features: 特征数量 (默认12)
        num_act_labels: 活动标签数量 (默认6)
        num_gen_labels: 性别标签数量 (默认1)
        label_codes: 活动标签编码字典
        trial_codes: 试验编码字典
        window_size: 窗口大小（时间步长）
        overlap: 窗口重叠比例
        normalize: 是否标准化
    
    Returns:
        train_features, test_features, train_labels, test_labels
        格式: [N, C, T] for features, [N, labels] for labels
    """
    # 默认配置
    if label_codes is None:
        label_codes = {
            "dws": num_features, "ups": num_features+1, "wlk": num_features+2,
            "jog": num_features+3, "sit": num_features+4, "std": num_features+5
        }
    
    if trial_codes is None:
        trial_codes = {
            "dws": [1,2,11], "ups": [3,4,12], "wlk": [7,8,15],
            "jog": [9,16], "sit": [5,13], "std": [6,14]
        }
    
    print(f"使用滑动窗口参数: window_size={window_size}, overlap={overlap}")
    
    dataset_columns = num_features + num_act_labels + num_gen_labels
    ds_list = get_ds_infos_windowed(data_dir)
    
    train_windows = []
    test_windows = []
    
    step_size = int(window_size * (1 - overlap))
    
    for i, sub_id in enumerate(ds_list[:, 0]):
        for j, act in enumerate(label_codes):
            for trial in trial_codes[act]:
                fname = os.path.join(
                    data_dir, 
                    "A_DeviceMotion_data", 
                    f"{act}_{trial}", 
                    f"sub_{int(sub_id)}.csv"
                )
                
                try:
                    raw_data = pd.read_csv(fname)
                    if 'Unnamed: 0' in raw_data.columns:
                        raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                    
                    if len(raw_data) < window_size:
                        print(f"跳过文件 {fname}，数据长度 {len(raw_data)} < 窗口大小 {window_size}")
                        continue
                    
                    unlabel_data = raw_data.values
                    
                    # 创建完整的标签数据
                    label_data = np.zeros((len(unlabel_data), dataset_columns))
                    label_data[:, :-(num_act_labels + num_gen_labels)] = unlabel_data
                    label_data[:, label_codes[act]] = 1
                    label_data[:, -(num_gen_labels)] = int(ds_list[i, 4])
                    
                    # 滑动窗口分割
                    for start_idx in range(0, len(label_data) - window_size + 1, step_size):
                        window = label_data[start_idx:start_idx + window_size]
                        
                        # 将数据分为特征和标签
                        features = window[:, :num_features]  # [T, C]
                        activity_label = window[0, num_features:num_features+num_act_labels]
                        gender_label = window[0, -num_gen_labels]
                        
                        # 转换为PyTorch格式 [C, T]
                        features_transposed = features.T
                        
                        # 组合标签信息
                        combined_labels = np.concatenate([activity_label, [gender_label]])
                        
                        # 根据trial划分训练/测试集
                        if trial > 10:
                            test_windows.append((features_transposed, combined_labels))
                        else:
                            train_windows.append((features_transposed, combined_labels))
                            
                except Exception as e:
                    print(f"处理文件 {fname} 时出错: {e}")
                    continue
    
    if not train_windows and not test_windows:
        raise ValueError("未能生成任何有效窗口")
    
    # 分离特征和标签
    if train_windows:
        train_features = np.array([w[0] for w in train_windows])  # [N, C, T]
        train_labels = np.array([w[1] for w in train_windows])    # [N, labels]
    else:
        train_features = np.empty((0, num_features, window_size))
        train_labels = np.empty((0, num_act_labels + num_gen_labels))
    
    if test_windows:
        test_features = np.array([w[0] for w in test_windows])   # [N, C, T]
        test_labels = np.array([w[1] for w in test_windows])     # [N, labels]
    else:
        test_features = np.empty((0, num_features, window_size))
        test_labels = np.empty((0, num_act_labels + num_gen_labels))
    
    # 标准化特征数据
    if normalize and len(train_features) > 0:
        print("执行数据标准化...")
        scaler = StandardScaler()
        
        # 训练集标准化
        N_train, C, T = train_features.shape
        train_reshaped = train_features.reshape(-1, C)
        train_normalized = scaler.fit_transform(train_reshaped)
        train_features = train_normalized.reshape(N_train, C, T)
        
        # 测试集使用相同的scaler
        if len(test_features) > 0:
            N_test, C, T = test_features.shape
            test_reshaped = test_features.reshape(-1, C)
            test_normalized = scaler.transform(test_reshaped)
            test_features = test_normalized.reshape(N_test, C, T)
    
    print(f"生成窗口统计:")
    print(f"  训练集: {len(train_features)} 个窗口")
    print(f"  测试集: {len(test_features)} 个窗口")
    print(f"  特征形状: [N, C={num_features}, T={window_size}]")
    
    return train_features, test_features, train_labels, test_labels

def save_motionsense_windowed_data(
    train_features: np.ndarray,
    test_features: np.ndarray, 
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    save_dir: str = 'motionsense_windowed_npy'
) -> None:
    """
    保存MotionSense窗口化数据为npy格式
    
    Args:
        train_features: 训练集特征 [N, C, T]
        test_features: 测试集特征 [N, C, T]
        train_labels: 训练集标签 [N, num_labels]
        test_labels: 测试集标签 [N, num_labels]
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存特征数据
    np.save(os.path.join(save_dir, 'X_train.npy'), train_features)
    np.save(os.path.join(save_dir, 'X_test.npy'), test_features)
    
    # 分离活动标签和性别标签
    train_activity_labels = np.argmax(train_labels[:, :6], axis=1)  # 活动标签 (0-5)
    test_activity_labels = np.argmax(test_labels[:, :6], axis=1)
    
    train_gender_labels = train_labels[:, 6].astype(int)  # 性别标签
    test_gender_labels = test_labels[:, 6].astype(int)
    
    # 保存标签
    np.save(os.path.join(save_dir, 'y_train.npy'), train_activity_labels)
    np.save(os.path.join(save_dir, 'y_test.npy'), test_activity_labels)
    np.save(os.path.join(save_dir, 'gender_train.npy'), train_gender_labels)
    np.save(os.path.join(save_dir, 'gender_test.npy'), test_gender_labels)
    
    # 创建验证集（使用测试集的副本）
    np.save(os.path.join(save_dir, 'X_val.npy'), test_features)
    np.save(os.path.join(save_dir, 'y_val.npy'), test_activity_labels)
    
    # 保存数据信息
    info = {
        'num_features': train_features.shape[1],
        'window_size': train_features.shape[2],
        'num_classes': 6,
        'train_samples': train_features.shape[0],
        'test_samples': test_features.shape[0],
        'activity_names': ["dws", "ups", "wlk", "jog", "sit", "std"]
    }
    
    np.save(os.path.join(save_dir, 'dataset_info.npy'), info)
    
    print(f"✅ MotionSense窗口化数据已保存到 {save_dir}/ 目录")
    print(f"   特征形状: 训练集 {train_features.shape}, 测试集 {test_features.shape}")
    print(f"   活动标签: 训练集 {train_activity_labels.shape}, 测试集 {test_activity_labels.shape}")
    print(f"   性别标签: 训练集 {train_gender_labels.shape}, 测试集 {test_gender_labels.shape}")

# 默认配置常量
MOTIONSENSE_CONFIG = {
    'num_features': 12,
    'num_act_labels': 6,
    'num_gen_labels': 1,
    'window_size': 128,
    'overlap': 0.5,
    'activity_names': ["dws", "ups", "wlk", "jog", "sit", "std"],
    'label_codes': {"dws": 12, "ups": 13, "wlk": 14, "jog": 15, "sit": 16, "std": 17},
    'trial_codes': {"dws": [1,2,11], "ups": [3,4,12], "wlk": [7,8,15], 
                   "jog": [9,16], "sit": [5,13], "std": [6,14]}
}
