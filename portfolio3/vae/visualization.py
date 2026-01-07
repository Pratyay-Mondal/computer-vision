"""
Visualization utilities for VAE.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

from model import VAE


def denormalize(tensor: torch.Tensor, normalized_to_minus_one: bool = True) -> torch.Tensor:
    """
    Denormalize tensor for visualization.
    
    Args:
        tensor: Input tensor
        normalized_to_minus_one: If True, denormalize from [-1, 1] to [0, 1]
                                  If False, just clamp to [0, 1]
    """
    if normalized_to_minus_one:
        return tensor * 0.5 + 0.5
    return torch.clamp(tensor, 0, 1)


def visualize_reconstructions(
    model: VAE,
    test_loader: DataLoader,
    device: torch.device,
    num_images: int = 5,
    title: str = "Reconstructions",
    save_path: Optional[Path] = None
):
    """Visualize original vs reconstructed images."""
    print(f"\n[VIZ] Generating {title}...")
    model.eval()
    
    normalized_to_minus_one = model.likelihood == "gaussian"
    
    with torch.no_grad():
        data = next(iter(test_loader)).to(device)[:num_images]
        recon, _, _ = model(data)
        
        data_display = denormalize(data, normalized_to_minus_one)
        recon_display = denormalize(recon, normalized_to_minus_one)
        
        comparison = torch.cat([data_display, recon_display])
        grid = utils.make_grid(comparison.cpu(), nrow=num_images, padding=2)
        
        plt.figure(figsize=(15, 6))
        plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
        plt.title(f"{title}\nTop: Original | Bottom: Reconstructed", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[VIZ] Saved to {save_path}")
        
        plt.show()


def latent_interpolation(
    model: VAE,
    test_loader: DataLoader,
    device: torch.device,
    num_sequences: int = 3,
    num_steps: int = 10,
    save_dir: Optional[Path] = None,
    save_prefix: str = "interpolation"
):
    """Perform latent space interpolation between pairs of images."""
    print(f"\n[VIZ] Generating {num_sequences} latent interpolation sequences...")
    model.eval()
    
    normalized_to_minus_one = model.likelihood == "gaussian"
    
    with torch.no_grad():
        data = next(iter(test_loader)).to(device)
        
        for i in range(num_sequences):
            img_pair = data[i*2:i*2+2]
            mu, _ = model.encode(img_pair)
            z1, z2 = mu[0], mu[1]
            
            alphas = torch.linspace(0, 1, num_steps).to(device)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                img_interp = model.decode(z_interp.unsqueeze(0))
                interpolations.append(img_interp)
            
            interpolations = torch.cat(interpolations)
            interpolations = denormalize(interpolations, normalized_to_minus_one)
            
            grid = utils.make_grid(interpolations.cpu(), nrow=num_steps, padding=2)
            
            plt.figure(figsize=(20, 3))
            plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
            plt.title(f"Latent Interpolation Sequence {i+1}", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            
            if save_dir:
                save_path = save_dir / f"{save_prefix}_seq{i+1}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[VIZ] Saved sequence {i+1} to {save_path}")
            
            plt.show()


def latent_traversal(
    model: VAE,
    test_loader: DataLoader,
    device: torch.device,
    num_dims: int = 5,
    num_steps: int = 10,
    traversal_range: float = 3.0,
    save_dir: Optional[Path] = None,
    save_prefix: str = "traversal"
):
    """Traverse individual latent dimensions."""
    print(f"\n[VIZ] Generating latent traversals for {num_dims} dimensions...")
    model.eval()
    
    normalized_to_minus_one = model.likelihood == "gaussian"
    
    with torch.no_grad():
        data = next(iter(test_loader)).to(device)[:1]
        mu, _ = model.encode(data)
        
        for dim in range(num_dims):
            z_traverse = mu.repeat(num_steps, 1)
            z_traverse[:, dim] = torch.linspace(
                -traversal_range, traversal_range, num_steps
            ).to(device)
            
            imgs = model.decode(z_traverse)
            imgs = denormalize(imgs, normalized_to_minus_one)
            
            grid = utils.make_grid(imgs.cpu(), nrow=num_steps, padding=2)
            
            plt.figure(figsize=(20, 3))
            plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
            plt.title(f"Latent Dimension {dim} Traversal "
                     f"(range: [{-traversal_range}, {traversal_range}])", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            
            if save_dir:
                save_path = save_dir / f"{save_prefix}_dim{dim}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[VIZ] Saved dimension {dim} to {save_path}")
            
            plt.show()


def plot_training_curves(
    history: dict,
    title: str,
    save_path: Optional[Path] = None
):
    """Plot training curves."""
    print(f"\n[VIZ] Plotting training curves for {title}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Loss
    axes[0, 0].plot(history['train_total_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['test_total_loss'], label='Test', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[0, 1].plot(history['train_recon_loss'], label='Train', linewidth=2, color='green')
    axes[0, 1].plot(history['test_recon_loss'], label='Test', linewidth=2, color='lightgreen')
    axes[0, 1].set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL Loss
    axes[1, 0].plot(history['train_kl_loss'], label='Train', linewidth=2, color='orange')
    axes[1, 0].plot(history['test_kl_loss'], label='Test', linewidth=2, color='gold')
    axes[1, 0].set_title('KL Divergence Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epoch Times
    axes[1, 1].bar(range(len(history['epoch_times'])), history['epoch_times'], 
                   color='purple', alpha=0.7)
    axes[1, 1].set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved curves to {save_path}")
    
    plt.show()


def compare_models(
    models_dict: dict[str, VAE],
    test_loader: DataLoader,
    device: torch.device,
    num_images: int = 5,
    save_path: Optional[Path] = None
):
    """Compare reconstructions from multiple models side by side."""
    print(f"\n[VIZ] Comparing {len(models_dict)} models...")
    
    data = next(iter(test_loader)).to(device)[:num_images]
    
    # Determine normalization from first model
    first_model = next(iter(models_dict.values()))
    normalized_to_minus_one = first_model.likelihood == "gaussian"
    
    rows = [denormalize(data, normalized_to_minus_one)]
    labels = ['Original']
    
    for name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(data)
            rows.append(denormalize(recon, normalized_to_minus_one))
            labels.append(name)
    
    comparison = torch.cat(rows)
    grid = utils.make_grid(comparison.cpu(), nrow=num_images, padding=2)
    
    plt.figure(figsize=(15, 3 * (len(models_dict) + 1)))
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
    plt.title("Model Comparison\n" + " | ".join(labels), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved comparison to {save_path}")
    
    plt.show()


def generate_samples(
    model: VAE,
    device: torch.device,
    num_samples: int = 16,
    title: str = "Generated Samples",
    save_path: Optional[Path] = None
):
    """Generate samples from the prior."""
    print(f"\n[VIZ] Generating {num_samples} samples from prior...")
    model.eval()
    
    normalized_to_minus_one = model.likelihood == "gaussian"
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
        samples = denormalize(samples, normalized_to_minus_one)
        
        nrow = int(np.sqrt(num_samples))
        grid = utils.make_grid(samples.cpu(), nrow=nrow, padding=2)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[VIZ] Saved samples to {save_path}")
        
        plt.show()
