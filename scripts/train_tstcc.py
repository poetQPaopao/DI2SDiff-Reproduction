#!/usr/bin/env python3
"""
TS-TCC Training Script

Train TS-TCC model on MotionSense or PAMAP2 dataset with self-supervised learning.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.motionsense_loader import create_windowed_time_series, MOTIONSENSE_CONFIG
from src.data.pamap2_loader import load_pamap2_data, PAMAP2_CONFIG
from src.models.tstcc import TSTCCModel, TemporalContrastiveModel, NTXentLoss, apply_augmentations
from configs.motionsense_config import MotionSenseConfig
from configs.pamap2_config import PAMAP2Config

def setup_logging(log_dir: str, experiment_name: str):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_dataset(dataset_name: str, data_dir: str = None):
    """加载数据集"""
    if dataset_name.lower() == 'motionsense':
        config = MotionSenseConfig()
        data_dir = data_dir or config.DATA_DIR
        
        train_features, test_features, train_labels, test_labels = create_windowed_time_series(
            data_dir=data_dir,
            num_features=config.NUM_FEATURES,
            num_act_labels=config.NUM_CLASSES,
            window_size=config.WINDOW_SIZE,
            overlap=config.OVERLAP,
            normalize=True
        )
        
        # 提取活动标签
        y_train = np.argmax(train_labels[:, :config.NUM_CLASSES], axis=1)
        y_test = np.argmax(test_labels[:, :config.NUM_CLASSES], axis=1)
        
        return train_features, test_features, y_train, y_test, config
        
    elif dataset_name.lower() == 'pamap2':
        config = PAMAP2Config()
        data_dir = data_dir or config.DATA_DIR
        
        X_train, X_test, y_train, y_test, train_subjects, test_subjects = load_pamap2_data(
            data_dir=data_dir,
            window_size=config.WINDOW_SIZE,
            overlap=config.OVERLAP,
            normalize=True,
            test_size=0.2
        )
        
        return X_train, X_test, y_train, y_test, config
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64):
    """创建数据加载器"""
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_self_supervised(model, temporal_model, train_loader, device, config, logger):
    """自监督训练"""
    model.train()
    temporal_model.train()
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(temporal_model.parameters()),
        lr=config.TSTCCConfig.LEARNING_RATE,
        weight_decay=config.TSTCCConfig.WEIGHT_DECAY
    )
    
    contrastive_loss_fn = NTXentLoss(temperature=config.TSTCCConfig.TEMPERATURE)
    
    train_losses = []
    
    for epoch in range(config.TSTCCConfig.NUM_EPOCHS):
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.TSTCCConfig.NUM_EPOCHS}')
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(device)
            
            # 数据增强
            aug1, aug2 = apply_augmentations(data, config.TSTCCConfig.AUGMENTATIONS)
            
            # 前向传播
            _, features1 = model(aug1)
            _, features2 = model(aug2)
            
            # 获取对比特征
            z1 = temporal_model(features1)
            z2 = temporal_model(features2)
            
            # 计算对比损失
            loss = contrastive_loss_fn(z1, z2)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        logger.info(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
    
    return train_losses

def evaluate_model(model, test_loader, device, logger):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            predictions, _ = model(data)
            
            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f'Test Accuracy: {accuracy:.4f}')
    logger.info('\nClassification Report:')
    logger.info(classification_report(all_labels, all_preds))
    
    return accuracy, all_preds, all_labels

def fine_tune_model(model, train_loader, test_loader, device, config, logger):
    """微调模型用于分类"""
    model.train()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.TSTCCConfig.LEARNING_RATE * 0.1,  # 较小的学习率
        weight_decay=config.TSTCCConfig.WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    
    for epoch in range(config.TSTCCConfig.NUM_EPOCHS // 2):  # 较少的epoch
        model.train()
        epoch_loss = 0.0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            predictions, _ = model(data)
            loss = criterion(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 评估
        accuracy, _, _ = evaluate_model(model, test_loader, device, logger)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            logger.info(f'New best accuracy: {best_accuracy:.4f}')
    
    return best_accuracy

def save_model(model, temporal_model, save_path, epoch, loss):
    """保存模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'temporal_model_state_dict': temporal_model.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def plot_training_curves(train_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TS-TCC Training Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train TS-TCC model')
    parser.add_argument('--dataset', type=str, choices=['motionsense', 'pamap2'], 
                       default='motionsense', help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default=None, 
                       help='Data directory path')
    parser.add_argument('--training_mode', type=str, 
                       choices=['self_supervised', 'fine_tune', 'supervised'], 
                       default='self_supervised', help='Training mode')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', 
                       help='Directory to save models')
    parser.add_argument('--experiment_name', type=str, default='tstcc_experiment', 
                       help='Experiment name')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 设置日志
    logger = setup_logging('./logs/', args.experiment_name)
    logger.info(f"Starting TS-TCC training experiment: {args.experiment_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Training mode: {args.training_mode}")
    logger.info(f"Device: {device}")
    
    # 加载数据集
    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test, config = load_dataset(args.dataset, args.data_dir)
    
    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Number of classes: {len(np.unique(y_train))}")
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        X_train, X_test, y_train, y_test, args.batch_size
    )
    
    # 创建模型
    if args.dataset.lower() == 'motionsense':
        model = TSTCCModel(
            input_channels=config.NUM_FEATURES,
            num_classes=config.NUM_CLASSES,
            mid_channels=64,
            final_out_channels=128
        ).to(device)
    else:  # pamap2
        model = TSTCCModel(
            input_channels=config.NUM_FEATURES,
            num_classes=config.NUM_CLASSES,
            mid_channels=64,
            final_out_channels=128
        ).to(device)
    
    temporal_model = TemporalContrastiveModel(
        input_dim=128,
        hidden_dim=100,
        output_dim=50
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    if args.training_mode == 'self_supervised':
        logger.info("Starting self-supervised training...")
        train_losses = train_self_supervised(
            model, temporal_model, train_loader, device, config, logger
        )
        
        # 保存模型
        save_path = os.path.join(args.save_dir, f"{args.experiment_name}_self_supervised.pth")
        save_model(model, temporal_model, save_path, args.epochs, train_losses[-1])
        
        # 绘制训练曲线
        plot_path = os.path.join('./results/', f"{args.experiment_name}_training_curve.png")
        plot_training_curves(train_losses, plot_path)
        
        # 微调用于分类
        logger.info("Starting fine-tuning for classification...")
        best_accuracy = fine_tune_model(model, train_loader, test_loader, device, config, logger)
        
        # 保存微调后的模型
        save_path = os.path.join(args.save_dir, f"{args.experiment_name}_fine_tuned.pth")
        save_model(model, temporal_model, save_path, args.epochs, best_accuracy)
    
    elif args.training_mode == 'supervised':
        # 直接监督学习训练
        logger.info("Starting supervised training...")
        # 这里可以添加监督学习的训练代码
        pass
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()
