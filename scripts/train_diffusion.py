#!/usr/bin/env python3
"""
Diffusion Model Training Script

Train 1D diffusion model for time-series generation on MotionSense or PAMAP2 dataset.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.motionsense_loader import create_windowed_time_series
from src.data.pamap2_loader import load_pamap2_data
from configs.motionsense_config import MotionSenseConfig
from configs.pamap2_config import PAMAP2Config

class UNet1D(nn.Module):
    """1D U-Net for time series diffusion"""
    
    def __init__(self, input_channels: int, time_emb_dim: int = 128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Encoder
        self.enc1 = self._make_conv_block(input_channels, 64)
        self.enc2 = self._make_conv_block(64, 128)
        self.enc3 = self._make_conv_block(128, 256)
        self.enc4 = self._make_conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self._make_conv_block(1024 + 512, 512)
        self.dec3 = self._make_conv_block(512 + 256, 256)
        self.dec2 = self._make_conv_block(256 + 128, 128)
        self.dec1 = self._make_conv_block(128 + 64, 64)
        
        # Output
        self.output = nn.Conv1d(64, input_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def positional_encoding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal positional encoding for timesteps"""
        half_dim = self.time_emb_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.positional_encoding(timesteps)
        t_emb = self.time_mlp(t_emb)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Output
        output = self.output(d1)
        return output

class DiffusionModel(nn.Module):
    """Denoising Diffusion Probabilistic Model for time series"""
    
    def __init__(self, unet: nn.Module, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Training loss computation"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.unet(x_noisy, t)
        
        loss = nn.MSELoss()(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single denoising step: p(x_{t-1} | x_t)"""
        betas_t = self.betas[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1)
        
        # Predict noise
        predicted_noise = self.unet(x, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t.min() == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], device: str = 'cpu') -> torch.Tensor:
        """Generate samples using reverse diffusion"""
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        
        return x

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
        
        return train_features, test_features, train_labels, test_labels, config
        
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

def train_diffusion_model(model: DiffusionModel, dataloader: DataLoader, device: str, 
                         num_epochs: int, learning_rate: float, logger):
    """训练扩散模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    for param_group in model.named_parameters():
        param_group[1].to(device)
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, model.timesteps, (batch_size,), device=device).long()
            
            # Compute loss
            loss = model.p_losses(data, t)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        logger.info(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
        
        # Sample and save generated data every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Generate samples
                samples = model.sample((4, data.shape[1], data.shape[2]), device=device)
                
                # Save sample plot
                save_sample_plot(samples.cpu().numpy(), epoch + 1, './results/')
    
    return train_losses

def save_sample_plot(samples: np.ndarray, epoch: int, save_dir: str):
    """保存生成样本的图片"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(4, len(samples))):
        # Plot first 3 channels of the time series
        for ch in range(min(3, samples.shape[1])):
            axes[i].plot(samples[i, ch, :], label=f'Channel {ch+1}')
        axes[i].set_title(f'Generated Sample {i+1}')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'generated_samples_epoch_{epoch}.png'))
    plt.close()

def evaluate_generation_quality(model: DiffusionModel, test_data: torch.Tensor, 
                               device: str, logger):
    """评估生成质量"""
    model.eval()
    
    with torch.no_grad():
        # Generate samples
        num_samples = min(100, len(test_data))
        generated_samples = model.sample(
            (num_samples, test_data.shape[1], test_data.shape[2]), 
            device=device
        )
        
        # Compute statistics
        real_mean = torch.mean(test_data[:num_samples], dim=0)
        real_std = torch.std(test_data[:num_samples], dim=0)
        
        gen_mean = torch.mean(generated_samples, dim=0)
        gen_std = torch.std(generated_samples, dim=0)
        
        # Mean absolute error between statistics
        mean_mae = torch.mean(torch.abs(real_mean - gen_mean)).item()
        std_mae = torch.mean(torch.abs(real_std - gen_std)).item()
        
        logger.info(f'Generation Quality Metrics:')
        logger.info(f'Mean MAE: {mean_mae:.4f}')
        logger.info(f'Std MAE: {std_mae:.4f}')
        
        return mean_mae, std_mae, generated_samples.cpu().numpy()

def save_model(model: DiffusionModel, save_path: str, epoch: int, loss: float):
    """保存模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'timesteps': model.timesteps,
        'betas': model.betas,
        'alphas_cumprod': model.alphas_cumprod
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Diffusion model')
    parser.add_argument('--dataset', type=str, choices=['motionsense', 'pamap2'], 
                       default='motionsense', help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default=None, 
                       help='Data directory path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', 
                       help='Directory to save models')
    parser.add_argument('--experiment_name', type=str, default='diffusion_experiment', 
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
    logger.info(f"Starting Diffusion training experiment: {args.experiment_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    
    # 加载数据集
    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test, config = load_dataset(args.dataset, args.data_dir)
    
    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 创建模型
    unet = UNet1D(input_channels=X_train.shape[1])
    diffusion_model = DiffusionModel(unet, timesteps=args.timesteps)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in diffusion_model.parameters())}")
    
    # 训练模型
    logger.info("Starting training...")
    train_losses = train_diffusion_model(
        diffusion_model, train_loader, device, args.epochs, args.lr, logger
    )
    
    # 评估生成质量
    logger.info("Evaluating generation quality...")
    test_tensor = torch.FloatTensor(X_test)
    mean_mae, std_mae, generated_samples = evaluate_generation_quality(
        diffusion_model, test_tensor, device, logger
    )
    
    # 保存模型
    save_path = os.path.join(args.save_dir, f"{args.experiment_name}.pth")
    save_model(diffusion_model, save_path, args.epochs, train_losses[-1])
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diffusion Model Training Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join('./results/', f"{args.experiment_name}_training_curve.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()
