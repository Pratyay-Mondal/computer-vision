#!/usr/bin/env python3
"""
Attribute Direction Experiment

Computes semantic directions in latent space based on CelebA attribute labels.
For example, finds the "Smiling" direction and visualizes traversal along it.

Usage:
    python -m experiments.attribute_direction --ckpt outputs/vae_best.pt --attribute Smiling
    python -m experiments.attribute_direction --ckpt outputs/vae_best.pt --attribute Eyeglasses --range 4.0
    python -m experiments.attribute_direction --ckpt outputs/vae_best.pt --list-attributes

Available attributes include:
    Smiling, Eyeglasses, Male, Young, Bald, Bangs, Black_Hair, Blond_Hair,
    Brown_Hair, Gray_Hair, Mustache, No_Beard, Wearing_Hat, Heavy_Makeup, etc.
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
from experiments.utils import (
    compute_attribute_direction,
    apply_direction,
    save_image_strip,
    get_value_range
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and visualize attribute directions in latent space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--attribute", type=str, default="Smiling",
        help="CelebA attribute to compute direction for"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="./dataset/img_align_celeba",
        help="Path to CelebA img_align_celeba directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/attributes",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-samples", type=int, default=5000,
        help="Maximum samples to use for computing direction"
    )
    parser.add_argument(
        "--steps", type=int, default=9,
        help="Number of steps in traversal"
    )
    parser.add_argument(
        "--range", type=float, default=3.0,
        help="Range for alpha values (-range to +range)"
    )
    parser.add_argument(
        "--num-examples", type=int, default=5,
        help="Number of example images to generate traversals for"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--list-attributes", action="store_true",
        help="List available attributes and exit"
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
    
    # Load model to get config
    print(f"[INFO] Loading model from {args.ckpt}")
    model, checkpoint = load_model(Path(args.ckpt), device)
    model.eval()
    
    config = checkpoint.get('config', {})
    likelihood = config.get('likelihood', 'gaussian')
    normalize_to_minus_one = likelihood == "gaussian"
    value_range = get_value_range(likelihood)
    
    print(f"[INFO] Model likelihood: {likelihood}")
    
    # Create dataset with attributes
    dataset = CelebADataset(
        root_dir=Path(args.dataset_dir),
        normalize_to_minus_one=normalize_to_minus_one,
        return_attributes=True
    )
    
    # List attributes if requested
    if args.list_attributes:
        print("\n[INFO] Available attributes:")
        for i, name in enumerate(dataset.get_attribute_names()):
            print(f"  {i+1:2d}. {name}")
        return
    
    # Check if attribute exists
    attr_names = dataset.get_attribute_names()
    if args.attribute not in attr_names:
        print(f"[ERROR] Attribute '{args.attribute}' not found!")
        print(f"[INFO] Available attributes: {', '.join(attr_names[:10])}...")
        return
    
    print(f"[INFO] Computing direction for attribute: {args.attribute}")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda"
    )
    
    # Collect latent codes and labels
    print(f"[INFO] Encoding up to {args.max_samples} samples...")
    latent_codes = []
    labels = []
    collected = 0
    
    with torch.no_grad():
        for batch in loader:
            images, attrs = batch
            images = images.to(device)
            
            # Encode
            mu, _ = model.encode(images)
            latent_codes.append(mu.cpu())
            
            # Get attribute labels
            # batch_labels = torch.tensor([
            #     attr.get(args.attribute, 0) for attr in attrs
            # ])
            batch_labels = attrs[args.attribute]
            labels.append(batch_labels)
            
            collected += images.size(0)
            if collected >= args.max_samples:
                break
            
            if collected % 1000 == 0:
                print(f"  Encoded {collected}/{args.max_samples} samples...")
    
    latent_codes = torch.cat(latent_codes, dim=0)[:args.max_samples]
    labels = torch.cat(labels, dim=0)[:args.max_samples]
    
    # Print statistics
    pos_count = (labels == 1).sum().item()
    neg_count = (labels == 0).sum().item()
    print(f"[INFO] Collected {len(labels)} samples: {pos_count} positive, {neg_count} negative")
    
    # Compute direction
    print(f"[INFO] Computing attribute direction...")
    direction = compute_attribute_direction(latent_codes, labels)
    
    # Save direction
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    direction_path = output_dir / f"direction_{args.attribute.lower()}.pt"
    torch.save({
        'direction': direction,
        'attribute': args.attribute,
        'num_samples': len(labels),
        'pos_count': pos_count,
        'neg_count': neg_count
    }, direction_path)
    print(f"[SAVED] Direction saved to {direction_path}")
    
    # Generate traversal visualizations
    print(f"\n[INFO] Generating {args.num_examples} traversal examples...")
    
    # Create test loader without attributes
    test_dataset = CelebADataset(
        root_dir=Path(args.dataset_dir),
        normalize_to_minus_one=normalize_to_minus_one,
        return_attributes=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    test_iter = iter(test_loader)
    
    alphas = torch.linspace(-args.range, args.range, args.steps).to(device)
    direction = direction.to(device)
    
    for i in range(args.num_examples):
        # Get a test image
        test_image = next(test_iter).to(device)
        
        with torch.no_grad():
            # Encode
            mu, _ = model.encode(test_image)
            z_ref = mu[0]
            
            # Apply direction at different scales
            z_modified = apply_direction(z_ref, direction, alphas)
            
            # Decode
            reconstructions = model.decode(z_modified)
        
        # Save strip
        out_path = output_dir / f"{args.attribute.lower()}_example{i+1}.png"
        save_image_strip(
            reconstructions.cpu(),
            out_path,
            normalize=True,
            value_range=value_range
        )
    
    print(f"\n[COMPLETE] Results saved to {output_dir}")
    print(f"  Direction: {direction_path.name}")
    print(f"  Traversals: {args.attribute.lower()}_example*.png")


if __name__ == "__main__":
    main()
