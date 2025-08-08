#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluate trained TS-TCC and Diffusion models on MotionSense or PAMAP2 dataset.
Includes classification metrics, representation quality, and generation evaluation.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from typing import Dict, Tuple, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.motionsense_loader import create_windowed_time_series
from src.data.pamap2_loader import load_pamap2_data
from src.models.tstcc import TSTCCModel, TemporalContrastiveModel
from configs.motionsense_config import MotionSenseConfig
from configs.pamap2_config import PAMAP2Config

def setup_logging(log_dir: str, experiment_name: str):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}_evaluation.log")
    
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

def load_trained_model(model_path: str, dataset_config, device: str):
    """加载训练好的模型"""
    # 创建模型实例
    model = TSTCCModel(
        input_channels=dataset_config.NUM_FEATURES,
        num_classes=dataset_config.NUM_CLASSES,
        mid_channels=64,
        final_out_channels=128
    )
    
    temporal_model = TemporalContrastiveModel(
        input_dim=128,
        hidden_dim=100,
        output_dim=50
    )
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'temporal_model_state_dict' in checkpoint:
        temporal_model.load_state_dict(checkpoint['temporal_model_state_dict'])
    
    model.to(device)
    temporal_model.to(device)
    model.eval()
    temporal_model.eval()
    
    return model, temporal_model

def extract_features(model: nn.Module, dataloader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """提取模型特征"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc='Extracting features'):
            data = data.to(device)
            _, features = model(data)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

def evaluate_classification(model: nn.Module, dataloader: DataLoader, device: str, 
                          class_names: List[str], logger) -> Dict:
    """评估分类性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc='Evaluating classification'):
            data, labels = data.to(device), labels.to(device)
            predictions, _ = model(data)
            
            probs = torch.softmax(predictions, dim=1)
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 分类报告
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    logger.info(f'Classification Accuracy: {accuracy:.4f}')
    logger.info('\nClassification Report:')
    logger.info(classification_report(all_labels, all_preds, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': class_report,
        'confusion_matrix': cm
    }

def evaluate_representation_quality(features: np.ndarray, labels: np.ndarray, 
                                  class_names: List[str], logger) -> Dict:
    """评估表示学习质量"""
    # 线性分类器评估
    lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
    svm_classifier = SVC(random_state=42)
    
    # 训练测试分割（使用前80%训练，后20%测试）
    split_idx = int(0.8 * len(features))
    
    X_train_feat, X_test_feat = features[:split_idx], features[split_idx:]
    y_train_feat, y_test_feat = labels[:split_idx], labels[split_idx:]
    
    # 训练线性分类器
    lr_classifier.fit(X_train_feat, y_train_feat)
    svm_classifier.fit(X_train_feat, y_train_feat)
    
    # 预测
    lr_preds = lr_classifier.predict(X_test_feat)
    svm_preds = svm_classifier.predict(X_test_feat)
    
    # 计算准确率
    lr_accuracy = accuracy_score(y_test_feat, lr_preds)
    svm_accuracy = accuracy_score(y_test_feat, svm_preds)
    
    logger.info(f'Linear Probe Accuracy (Logistic Regression): {lr_accuracy:.4f}')
    logger.info(f'Linear Probe Accuracy (SVM): {svm_accuracy:.4f}')
    
    return {
        'linear_probe_lr_accuracy': lr_accuracy,
        'linear_probe_svm_accuracy': svm_accuracy,
        'lr_predictions': lr_preds,
        'svm_predictions': svm_preds,
        'test_labels': y_test_feat
    }

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_tsne_visualization(features: np.ndarray, labels: np.ndarray, 
                           class_names: List[str], save_path: str):
    """绘制t-SNE可视化"""
    # 计算t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(class_names)), 
                label='Activity Class')
    plt.title('t-SNE Visualization of Learned Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # 添加图例
    handles = [plt.scatter([], [], c=plt.cm.tab10(i), label=class_names[i]) 
               for i in range(len(class_names))]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_accuracy(class_report: Dict, class_names: List[str], save_path: str):
    """绘制每类准确率"""
    precisions = [class_report[str(i)]['precision'] for i in range(len(class_names))]
    recalls = [class_report[str(i)]['recall'] for i in range(len(class_names))]
    f1_scores = [class_report[str(i)]['f1-score'] for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    plt.bar(x, recalls, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Activity Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Classification Metrics')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def generate_evaluation_report(results: Dict, dataset_name: str, model_name: str, 
                             save_path: str):
    """生成评估报告"""
    report = f"""
# {model_name} Evaluation Report on {dataset_name.upper()} Dataset

## Overall Performance
- **Classification Accuracy**: {results['classification']['accuracy']:.4f}
- **Linear Probe Accuracy (LR)**: {results['representation']['linear_probe_lr_accuracy']:.4f}
- **Linear Probe Accuracy (SVM)**: {results['representation']['linear_probe_svm_accuracy']:.4f}

## Per-Class Metrics
"""
    
    class_report = results['classification']['classification_report']
    for i, class_name in enumerate(results['class_names']):
        if str(i) in class_report:
            precision = class_report[str(i)]['precision']
            recall = class_report[str(i)]['recall']
            f1 = class_report[str(i)]['f1-score']
            support = class_report[str(i)]['support']
            
            report += f"- **{class_name}**: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}\n"
    
    report += f"""
## Macro Averages
- **Precision**: {class_report['macro avg']['precision']:.4f}
- **Recall**: {class_report['macro avg']['recall']:.4f}
- **F1-Score**: {class_report['macro avg']['f1-score']:.4f}

## Weighted Averages
- **Precision**: {class_report['weighted avg']['precision']:.4f}
- **Recall**: {class_report['weighted avg']['recall']:.4f}
- **F1-Score**: {class_report['weighted avg']['f1-score']:.4f}
"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--dataset', type=str, choices=['motionsense', 'pamap2'], 
                       default='motionsense', help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default=None, 
                       help='Data directory path')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['tstcc', 'diffusion'],
                       default='tstcc', help='Type of model to evaluate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--results_dir', type=str, default='./evaluation_results/', 
                       help='Directory to save evaluation results')
    parser.add_argument('--experiment_name', type=str, default='evaluation', 
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
    logger.info(f"Starting model evaluation: {args.experiment_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model path: {args.model_path}")
    
    # 加载数据集
    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test, config = load_dataset(args.dataset, args.data_dir)
    
    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 获取类别名称
    if args.dataset.lower() == 'motionsense':
        class_names = ['downstairs', 'upstairs', 'walking', 'jogging', 'sitting', 'standing']
    else:  # pamap2
        class_names = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'rope_jumping', 'other']
    
    if args.model_type == 'tstcc':
        # 加载TS-TCC模型
        logger.info("Loading TS-TCC model...")
        model, temporal_model = load_trained_model(args.model_path, config, device)
        
        # 评估分类性能
        logger.info("Evaluating classification performance...")
        classification_results = evaluate_classification(
            model, test_loader, device, class_names, logger
        )
        
        # 提取特征
        logger.info("Extracting features...")
        train_features, train_labels = extract_features(model, train_loader, device)
        test_features, test_labels = extract_features(model, test_loader, device)
        
        # 评估表示学习质量
        logger.info("Evaluating representation quality...")
        representation_results = evaluate_representation_quality(
            test_features, test_labels, class_names, logger
        )
        
        # 生成可视化
        logger.info("Generating visualizations...")
        
        # 混淆矩阵
        plot_confusion_matrix(
            classification_results['confusion_matrix'], 
            class_names,
            os.path.join(args.results_dir, f"{args.experiment_name}_confusion_matrix.png")
        )
        
        # t-SNE可视化
        plot_tsne_visualization(
            test_features, test_labels, class_names,
            os.path.join(args.results_dir, f"{args.experiment_name}_tsne.png")
        )
        
        # 每类准确率
        plot_per_class_accuracy(
            classification_results['classification_report'], 
            class_names,
            os.path.join(args.results_dir, f"{args.experiment_name}_per_class_metrics.png")
        )
        
        # 生成报告
        results = {
            'classification': classification_results,
            'representation': representation_results,
            'class_names': class_names
        }
        
        generate_evaluation_report(
            results, args.dataset, 'TS-TCC',
            os.path.join(args.results_dir, f"{args.experiment_name}_report.md")
        )
    
    elif args.model_type == 'diffusion':
        logger.info("Diffusion model evaluation not implemented yet")
        # TODO: 实现扩散模型的评估
        pass
    
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to: {args.results_dir}")

if __name__ == '__main__':
    main()
