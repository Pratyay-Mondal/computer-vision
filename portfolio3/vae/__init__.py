"""
VAE Training Package for CelebA Dataset.

Supports multiple likelihood distributions:
- gaussian: MSE loss with Tanh output
- continuous_bernoulli: Continuous Bernoulli distribution
- bernoulli: Standard BCE loss
"""

from .config import Config
from .model import VAE
from .losses import compute_vae_loss
from .trainer import VAETrainer, load_model
from .dataset import CelebADataset, create_dataloaders

__all__ = [
    "Config",
    "VAE",
    "compute_vae_loss",
    "VAETrainer",
    "load_model",
    "CelebADataset",
    "create_dataloaders",
]
