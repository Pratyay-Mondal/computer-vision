#!/usr/bin/env python3
"""
Latent Space Interpolation Experiment

Performs linear interpolation between pairs of images in the latent space.
This demonstrates the smoothness and structure of the learned latent space.

Usage:
    python -m experiments.interpolation --ckpt outputs/vae_best.pt
    python -m experiments.interpolation --ckpt outputs/vae_best.pt --pairs 5 --steps 15
"""

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import CelebADataset
from model import VAE
from trainer import load_model
from experiments.utils import save_image_strip, get_value_range


def interpolate_latent(z1: torch.Tensor, z2: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Linear interpolation between two latent vectors.
    
    Args:
        z1: First latent vector (latent_dim,)
        z2: Second latent vector (latent_dim,)
        steps: Number of interpolation steps
        
    Returns:
        Interpolated vectors (steps, latent_dim)
    """
    alphas = torch.linspace(0, 1, steps, device=z1.device)
    return torch.stack([(1 - a) * z1 + a * z2 for a in alphas], dim=0)


def spherical_interpolate(z1: torch.Tensor, z2: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Spherical linear interpolation (slerp) between two latent vectors.
    Better preserves magnitude, useful for VAEs with unit Gaussian prior.
    
    Args:
        z1: First latent vector (latent_dim,)
        z2: Second latent vector (latent_dim,)
        steps: Number of interpolation steps
        
    Returns:
        Interpolated vectors (steps, latent_dim)
    """
    # Normalize
    z1_norm = z1 / (z1.norm() + 1e-8)
    z2_norm = z2 / (z2.norm() + 1e-8)
    
    # Compute angle
    dot = (z1_norm * z2_norm).sum().clamp(-1, 1)
    theta = torch.acos(dot)
    
    # Handle near-parallel vectors
    if theta.abs() < 1e-6:
        return interpolate_latent(z1, z2, steps)
    
    alphas = torch.linspace(0, 1, steps, device=z1.device)
    
    result = []
    for a in alphas:
        w1 = torch.sin((1 - a) * theta) / torch.sin(theta)
        w2 = torch.sin(a * theta) / torch.sin(theta)
        z_interp = w1 * z1 + w2 * z2
        result.append(z_interp)
    
    return torch.stack(result, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform latent space interpolation between image pairs"
    )
    
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="./dataset/img_align_celeba",
        help="Path to CelebA img_align_celeba directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/interpolation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--pairs", type=int, default=5,
        help="Number of image pairs to interpolate"
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of interpolation steps"
    )
    parser.add_argument(
        "--method", type=str, default="linear",
        choices=["linear", "spherical"],
        help="Interpolation method"
    )
    parser.add_argument(
        "--include-originals", action="store_true",
        help="Include original images at the ends of the strip"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (auto-detected if not specified)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load model
    print(f"[INFO] Loading model from {args.ckpt}")
    model, checkpoint = load_model(Path(args.ckpt), device)
    model.eval()
    
    config = checkpoint.get('config', {})
    likelihood = config.get('likelihood', 'gaussian')
    normalize_to_minus_one = likelihood == "gaussian"
    value_range = get_value_range(likelihood)
    
    print(f"[INFO] Model likelihood: {likelihood}")
    print(f"[INFO] Interpolation method: {args.method}")
    
    # Create dataset
    dataset = CelebADataset(
        root_dir=Path(args.dataset_dir),
        normalize_to_minus_one=normalize_to_minus_one,
        return_attributes=False
    )
    
    # Dataloader returns pairs
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose interpolation function
    interp_fn = spherical_interpolate if args.method == "spherical" else interpolate_latent
    
    print(f"\n[INFO] Generating {args.pairs} interpolation sequences...")
    
    loader_iter = iter(loader)
    
    for pair_idx in range(args.pairs):
        # Get pair of images
        try:
            images = next(loader_iter).to(device)
        except StopIteration:
            loader_iter = iter(loader)
            images = next(loader_iter).to(device)
        
        with torch.no_grad():
            # Encode both images
            mu, _ = model.encode(images)
            z1, z2 = mu[0], mu[1]
            
            # Interpolate
            z_interp = interp_fn(z1, z2, args.steps)
            
            # Decode
            reconstructions = model.decode(z_interp)
            
            # Optionally include originals
            if args.include_originals:
                # Reconstruct originals for comparison
                orig_recon, _, _ = model(images)
                # Prepend first original, append second
                reconstructions = torch.cat([
                    orig_recon[0:1],
                    reconstructions,
                    orig_recon[1:2]
                ], dim=0)
        
        # Save strip
        out_path = output_dir / f"interp_pair{pair_idx + 1}.png"
        save_image_strip(
            reconstructions.cpu(),
            out_path,
            normalize=True,
            value_range=value_range
        )
    
    print(f"\n[COMPLETE] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
