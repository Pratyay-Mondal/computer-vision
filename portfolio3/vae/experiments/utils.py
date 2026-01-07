"""
Shared utilities for experiment scripts.
"""

from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import matplotlib.pyplot as plt


def denormalize(
    tensor: torch.Tensor, 
    normalized_to_minus_one: bool = True
) -> torch.Tensor:
    """
    Denormalize tensor for visualization.
    
    Args:
        tensor: Input tensor
        normalized_to_minus_one: If True, denormalize from [-1, 1] to [0, 1]
    """
    if normalized_to_minus_one:
        return tensor * 0.5 + 0.5
    return torch.clamp(tensor, 0, 1)


def save_image_strip(
    images: torch.Tensor,
    save_path: Path,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    padding: int = 2,
    pad_value: float = 1.0
):
    """
    Save a strip of images as a single image.
    
    Args:
        images: Tensor of shape (N, C, H, W)
        save_path: Path to save the image
        normalize: Whether to normalize to [0, 1]
        value_range: If provided, use this range for normalization
        padding: Padding between images
        pad_value: Value for padding pixels
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        if value_range:
            # Normalize from value_range to [0, 1]
            images = (images - value_range[0]) / (value_range[1] - value_range[0])
        images = torch.clamp(images, 0, 1)
    
    grid = vutils.make_grid(images, nrow=images.size(0), padding=padding, pad_value=pad_value)
    
    # Convert to numpy and save
    grid_np = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(2 * images.size(0), 2))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"[SAVED] {save_path}")


def save_image_grid(
    images: torch.Tensor,
    save_path: Path,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    labels: Optional[List[Tuple[str, Tuple[int, int]]]] = None,
    padding: int = 2
):
    """
    Save a grid of images.
    
    Args:
        images: Tensor of shape (N, C, H, W)
        save_path: Path to save the image
        nrow: Number of images per row
        normalize: Whether to normalize to [0, 1]
        value_range: If provided, use this range for normalization
        title: Optional title for the figure
        labels: Optional list of (text, (x, y)) tuples for labels
        padding: Padding between images
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        if value_range:
            images = (images - value_range[0]) / (value_range[1] - value_range[0])
        images = torch.clamp(images, 0, 1)
    
    grid = vutils.make_grid(images, nrow=nrow, padding=padding, pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).numpy()
    
    ncol = min(nrow, images.size(0))
    nrows_grid = (images.size(0) + nrow - 1) // nrow
    
    fig, ax = plt.subplots(figsize=(2 * ncol, 2 * nrows_grid))
    ax.imshow(grid_np)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=12)
    
    if labels:
        for text, (x, y) in labels:
            ax.text(x, y, text, fontsize=10, color='white', 
                   backgroundcolor='black', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {save_path}")


def compute_attribute_direction(
    latent_codes: torch.Tensor,
    attribute_labels: torch.Tensor,
    method: str = "difference"
) -> torch.Tensor:
    """
    Compute a direction in latent space corresponding to an attribute.
    
    Args:
        latent_codes: Tensor of shape (N, latent_dim)
        attribute_labels: Binary tensor of shape (N,) with 0/1 labels
        method: Method to compute direction ("difference" or "svm")
        
    Returns:
        Direction vector of shape (latent_dim,)
    """
    # Ensure labels are binary
    labels = (attribute_labels > 0).float()
    
    # Split by attribute
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        raise ValueError("Need both positive and negative samples for attribute")
    
    pos_codes = latent_codes[pos_mask]
    neg_codes = latent_codes[neg_mask]
    
    if method == "difference":
        # Simple difference of means
        pos_mean = pos_codes.mean(dim=0)
        neg_mean = neg_codes.mean(dim=0)
        direction = pos_mean - neg_mean
        # Normalize
        direction = direction / (direction.norm() + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return direction


def apply_direction(
    z: torch.Tensor,
    direction: torch.Tensor,
    alphas: torch.Tensor
) -> torch.Tensor:
    """
    Apply a direction to a latent code with multiple scales.
    
    Args:
        z: Base latent code of shape (latent_dim,)
        direction: Direction vector of shape (latent_dim,)
        alphas: Scale factors of shape (num_steps,)
        
    Returns:
        Modified latent codes of shape (num_steps, latent_dim)
    """
    # z: (D,), direction: (D,), alphas: (S,)
    # Result: (S, D)
    return z.unsqueeze(0) + alphas.unsqueeze(1) * direction.unsqueeze(0)


def get_value_range(likelihood: str) -> Tuple[float, float]:
    """Get the value range based on likelihood type."""
    if likelihood == "gaussian":
        return (-1.0, 1.0)
    else:
        return (0.0, 1.0)
