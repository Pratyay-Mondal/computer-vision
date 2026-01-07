"""
Loss functions for VAE training with different likelihood distributions.
"""

import torch
import torch.nn.functional as F
from torch.distributions import ContinuousBernoulli
from typing import Literal


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between approximate posterior and prior.
    
    KL(q(z|x) || p(z)) where q(z|x) = N(mu, sigma^2) and p(z) = N(0, I)
    
    Args:
        mu: Mean of approximate posterior
        logvar: Log variance of approximate posterior
        
    Returns:
        KL divergence (sum over all dimensions and batch)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def gaussian_reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian reconstruction loss (MSE).
    
    Corresponds to assuming p(x|z) = N(recon_x, I).
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        
    Returns:
        MSE loss (sum over all dimensions and batch)
    """
    return F.mse_loss(recon_x, x, reduction='sum')


def continuous_bernoulli_reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Continuous Bernoulli reconstruction loss.
    
    Better suited for continuous data in [0, 1] than standard Bernoulli.
    Includes the normalizing constant.
    
    Args:
        recon_x: Reconstructed images (probabilities in [0, 1])
        x: Original images (values in [0, 1])
        
    Returns:
        Negative log likelihood (sum over all dimensions and batch)
    """
    dist = ContinuousBernoulli(probs=recon_x)
    return -dist.log_prob(x).sum()


def bernoulli_reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Standard Bernoulli reconstruction loss (Binary Cross Entropy).
    
    Commonly used but technically incorrect for continuous data.
    
    Args:
        recon_x: Reconstructed images (probabilities in [0, 1])
        x: Original images (values in [0, 1])
        
    Returns:
        BCE loss (sum over all dimensions and batch)
    """
    return F.binary_cross_entropy(recon_x, x, reduction='sum')


def compute_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    likelihood: Literal["gaussian", "continuous_bernoulli", "bernoulli"] = "gaussian"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total VAE loss with specified likelihood.
    
    ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
    Loss = -ELBO = -E[log p(x|z)] + beta * KL
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of approximate posterior
        logvar: Log variance of approximate posterior
        beta: Weight for KL divergence (beta-VAE parameter)
        likelihood: Type of likelihood distribution
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss)
    """
    # Compute reconstruction loss based on likelihood type
    if likelihood == "gaussian":
        recon_loss = gaussian_reconstruction_loss(recon_x, x)
    elif likelihood == "continuous_bernoulli":
        recon_loss = continuous_bernoulli_reconstruction_loss(recon_x, x)
    elif likelihood == "bernoulli":
        recon_loss = bernoulli_reconstruction_loss(recon_x, x)
    else:
        raise ValueError(f"Unknown likelihood type: {likelihood}")
    
    # Compute KL divergence
    kl_loss = compute_kl_divergence(mu, logvar)
    
    # Total loss (negative ELBO)
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
