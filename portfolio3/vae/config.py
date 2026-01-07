"""
Configuration settings for VAE training on CelebA dataset.
"""

import torch
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path


@dataclass
class Config:
    """Configuration for VAE training."""
    
    # Paths
    dataset_dir: Path = field(default_factory=lambda: Path("./dataset/img_align_celeba"))
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    
    # Model hyperparameters
    batch_size: int = 128
    image_size: int = 64
    latent_dim: int = 128
    epochs: int = 10
    learning_rate: float = 1e-4
    dataset_limit: int = 200000
    
    # VAE configuration
    beta: float = 1.0
    likelihood: Literal["gaussian", "continuous_bernoulli", "bernoulli"] = "gaussian"
    
    # Training settings
    num_workers: int = 2
    pin_memory: bool = True
    train_split: float = 0.9
    
    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def __post_init__(self):
        """Convert paths to Path objects if needed."""
        if isinstance(self.dataset_dir, str):
            self.dataset_dir = Path(self.dataset_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def __str__(self):
        return (
            f"Configuration:\n"
            f"  Dataset Dir: {self.dataset_dir}\n"
            f"  Output Dir: {self.output_dir}\n"
            f"  Batch Size: {self.batch_size}\n"
            f"  Image Size: {self.image_size}x{self.image_size}\n"
            f"  Latent Dimension: {self.latent_dim}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Learning Rate: {self.learning_rate}\n"
            f"  Dataset Limit: {self.dataset_limit}\n"
            f"  Beta: {self.beta}\n"
            f"  Likelihood: {self.likelihood}\n"
            f"  Device: {self.device}"
        )
