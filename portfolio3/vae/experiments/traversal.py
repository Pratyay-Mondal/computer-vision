#!/usr/bin/env python3
"""
Latent Dimension Traversal Experiment

Traverses individual latent dimensions to visualize what each dimension encodes.
Can automatically select the most informative dimensions based on variance.

Usage:
    # Traverse top-16 most variable dimensions
    python -m experiments.traversal --ckpt outputs/vae_best.pt
    
    # Traverse specific dimensions
    python -m experiments.traversal --ckpt outputs/vae_best.pt --dims 0 5 10 15 20
    
    # Traverse first N dimensions (no variance selection)
    python -m experiments.traversal --ckpt outputs/vae_best.pt --num-dims 10 --no-variance-selection
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import CelebADataset
from model import VAE
from trainer import load_model
from experiments.utils import save_image_strip, save_image_grid, get_value_range


def compute_latent_statistics(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 2048
) -> dict:
    """
    Compute statistics of the latent space.
    
    Returns:
        Dictionary with 'mean', 'std', 'var' tensors of shape (latent_dim,)
    """
    model.eval()
    
    mus = []
    collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            
            mu, _ = model.encode(batch)
            mus.append(mu.cpu())
            
            collected += batch.size(0)
            if collected >= max_samples:
                break
    
    mu_all = torch.cat(mus, dim=0)[:max_samples]
    
    return {
        'mean': mu_all.mean(dim=0),
        'std': mu_all.std(dim=0),
        'var': mu_all.var(dim=0),
        'num_samples': len(mu_all)
    }


def select_dimensions(
    stats: dict,
    num_dims: int,
    method: str = "variance"
) -> List[int]:
    """
    Select which dimensions to traverse.
    
    Args:
        stats: Dictionary from compute_latent_statistics
        num_dims: Number of dimensions to select
        method: Selection method ("variance", "first")
        
    Returns:
        List of dimension indices
    """
    latent_dim = stats['std'].numel()
    num_dims = min(num_dims, latent_dim)
    
    if method == "variance":
        # Select dimensions with highest variance
        indices = torch.topk(stats['var'], k=num_dims).indices.tolist()
        return sorted(indices)
    else:
        # Just take first N
        return list(range(num_dims))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traverse latent dimensions with variance-based selection"
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
        "--output-dir", type=str, default="./outputs/traversal",
        help="Output directory for results"
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", default=None,
        help="Specific dimensions to traverse (overrides --num-dims)"
    )
    parser.add_argument(
        "--num-dims", type=int, default=16,
        help="Number of dimensions to traverse"
    )
    parser.add_argument(
        "--steps", type=int, default=9,
        help="Number of steps in each traversal"
    )
    parser.add_argument(
        "--range", type=float, default=3.0,
        help="Range for traversal (-range to +range)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=2048,
        help="Max samples for computing variance statistics"
    )
    parser.add_argument(
        "--no-variance-selection", action="store_true",
        help="Don't use variance-based selection, just take first N dims"
    )
    parser.add_argument(
        "--num-examples", type=int, default=3,
        help="Number of reference images to use"
    )
    parser.add_argument(
        "--combined-grid", action="store_true",
        help="Also save a combined grid of all dimensions"
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
    latent_dim = config.get('latent_dim', 128)
    normalize_to_minus_one = likelihood == "gaussian"
    value_range = get_value_range(likelihood)
    
    print(f"[INFO] Model likelihood: {likelihood}")
    print(f"[INFO] Latent dimension: {latent_dim}")
    
    # Create dataset
    dataset = CelebADataset(
        root_dir=Path(args.dataset_dir),
        normalize_to_minus_one=normalize_to_minus_one,
        return_attributes=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    
    # Determine which dimensions to traverse
    if args.dims:
        dim_indices = args.dims
        print(f"[INFO] Using specified dimensions: {dim_indices}")
    else:
        if args.no_variance_selection:
            dim_indices = list(range(min(args.num_dims, latent_dim)))
            print(f"[INFO] Using first {len(dim_indices)} dimensions")
        else:
            print(f"[INFO] Computing latent statistics for variance-based selection...")
            stats = compute_latent_statistics(model, loader, device, args.max_samples)
            dim_indices = select_dimensions(stats, args.num_dims, method="variance")
            
            print(f"[INFO] Selected {len(dim_indices)} dimensions by variance:")
            for i, dim in enumerate(dim_indices):
                print(f"  z{dim:3d}: std={stats['std'][dim]:.4f}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get reference images
    ref_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    ref_iter = iter(ref_loader)
    
    alphas = torch.linspace(-args.range, args.range, args.steps, device=device)
    
    print(f"\n[INFO] Generating traversals for {len(dim_indices)} dimensions x {args.num_examples} examples...")
    
    for example_idx in range(args.num_examples):
        # Get reference image
        ref_image = next(ref_iter).to(device)
        
        with torch.no_grad():
            mu, _ = model.encode(ref_image)
            z_ref = mu[0]
        
        all_traversals = []
        
        for dim in dim_indices:
            # Create traversal for this dimension
            z_batch = z_ref.unsqueeze(0).repeat(args.steps, 1)
            z_batch[:, dim] = z_ref[dim] + alphas
            
            with torch.no_grad():
                recon = model.decode(z_batch)
            
            # Save individual strip
            out_path = output_dir / f"example{example_idx+1}_z{dim:03d}.png"
            save_image_strip(
                recon.cpu(),
                out_path,
                normalize=True,
                value_range=value_range
            )
            
            all_traversals.append(recon.cpu())
        
        # Save combined grid if requested
        if args.combined_grid and len(all_traversals) > 0:
            # Stack all traversals: (num_dims, steps, C, H, W)
            combined = torch.stack(all_traversals, dim=0)
            # Reshape to (num_dims * steps, C, H, W)
            combined = combined.view(-1, *combined.shape[2:])
            
            grid_path = output_dir / f"example{example_idx+1}_all_dims.png"
            save_image_grid(
                combined,
                grid_path,
                nrow=args.steps,
                normalize=True,
                value_range=value_range,
                title=f"Latent Traversals (dims: {dim_indices})"
            )
    
    # Save dimension info
    info_path = output_dir / "traversal_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Dimensions traversed: {dim_indices}\n")
        f.write(f"Traversal range: [{-args.range}, {args.range}]\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Examples: {args.num_examples}\n")
        if not args.no_variance_selection and not args.dims:
            f.write(f"\nVariance-based selection used\n")
            f.write(f"Samples for statistics: {stats['num_samples']}\n")
    
    print(f"\n[COMPLETE] Results saved to {output_dir}")
    print(f"  Traversals: example*_z*.png")
    if args.combined_grid:
        print(f"  Combined grids: example*_all_dims.png")


if __name__ == "__main__":
    main()
