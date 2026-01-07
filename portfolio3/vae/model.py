"""
VAE Model Architecture with ResNet-style encoder and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class ResBlockDown(nn.Module):
    """Residual block with downsampling."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2)

    def forward(self, x):
        return F.leaky_relu(self.conv(x) + self.skip(x), 0.2)


class ResBlockUp(nn.Module):
    """Residual block with upsampling."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return F.relu(self.conv(x) + self.skip(x))


class VAE(nn.Module):
    """
    Variational Autoencoder with ResNet-style architecture.
    
    Supports multiple likelihood distributions:
    - gaussian: MSE loss with Tanh output (normalized to [-1, 1])
    - continuous_bernoulli: Continuous Bernoulli with Sigmoid output
    - bernoulli: Standard Bernoulli (BCE) with Sigmoid output
    """
    
    def __init__(
        self, 
        latent_dim: int = 128,
        likelihood: Literal["gaussian", "continuous_bernoulli", "bernoulli"] = "gaussian"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.likelihood = likelihood
        
        # Encoder: 3x64x64 -> 256x4x4
        self.enc = nn.Sequential(
            ResBlockDown(3, 32),    # 32x32
            ResBlockDown(32, 64),   # 16x16
            ResBlockDown(64, 128),  # 8x8
            ResBlockDown(128, 256)  # 4x4
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder: latent -> 256x4x4 -> 3x64x64
        self.dec_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Build decoder with appropriate output activation
        decoder_layers = [
            ResBlockUp(256, 128),   # 8x8
            ResBlockUp(128, 64),    # 16x16
            ResBlockUp(64, 32),     # 32x32
            ResBlockUp(32, 16),     # 64x64
            nn.Conv2d(16, 3, 3, padding=1),
        ]
        
        # Output activation based on likelihood
        if likelihood == "gaussian":
            decoder_layers.append(nn.Tanh())  # Output in [-1, 1]
        else:
            decoder_layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.dec = nn.Sequential(*decoder_layers)
        
        print(f"[MODEL] VAE initialized with {likelihood} likelihood")
        print(f"[MODEL] Latent dimension: {latent_dim}")
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.dec(self.dec_fc(z).view(z.size(0), 256, 4, 4))
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input images
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        # Clamp for numerical stability (Bernoulli likelihoods)
        if self.likelihood in ["continuous_bernoulli", "bernoulli"]:
            recon_x = torch.clamp(recon_x, 1e-6, 1 - 1e-6)
        
        return recon_x, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from the prior."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
