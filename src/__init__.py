"""
HAR-Diffusion-Project: Human Activity Recognition with Diffusion Models

This package provides comprehensive tools for human activity recognition using
state-of-the-art deep learning techniques including TS-TCC and Diffusion Models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data import motionsense_loader, pamap2_loader
from .models import tstcc, diffusion
from .training import tstcc_trainer, diffusion_trainer
from .evaluation import metrics, visualization

__all__ = [
    "motionsense_loader",
    "pamap2_loader", 
    "tstcc",
    "diffusion",
    "tstcc_trainer",
    "diffusion_trainer",
    "metrics",
    "visualization"
]
