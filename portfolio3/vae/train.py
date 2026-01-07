#!/usr/bin/env python3
"""
Main training script for VAE on CelebA dataset.

Usage:
    # Train baseline VAE with Gaussian likelihood (default)
    python train.py --beta 1.0
    
    # Train beta-VAE with higher beta
    python train.py --beta 4.0 --name beta_vae_4
    
    # Train with Continuous Bernoulli likelihood
    python train.py --likelihood continuous_bernoulli --name vae_cb
    
    # Train with standard Bernoulli likelihood
    python train.py --likelihood bernoulli --name vae_bernoulli
    
    # Full example with all options
    python train.py --beta 1.0 --likelihood gaussian --epochs 10 --batch-size 128 \
                    --latent-dim 128 --lr 1e-4 --dataset-limit 200000 --name my_vae
"""

import argparse
from pathlib import Path

import torch

from config import Config
from dataset import create_dataloaders
from model import VAE
from trainer import VAETrainer
from visualization import (
    visualize_reconstructions,
    latent_interpolation,
    latent_traversal,
    plot_training_curves,
    generate_samples
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VAE on CelebA dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Beta parameter for KL divergence weight (beta-VAE)"
    )
    parser.add_argument(
        "--likelihood", type=str, default="gaussian",
        choices=["gaussian", "continuous_bernoulli", "bernoulli"],
        help="Likelihood distribution for reconstruction loss"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=128,
        help="Dimension of latent space"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--dataset-limit", type=int, default=200000,
        help="Maximum number of images to use from dataset"
    )
    
    # Paths
    parser.add_argument(
        "--dataset-dir", type=str, default="./dataset/img_align_celeba",
        help="Path to CelebA img_align_celeba directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Directory for saving outputs"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Name for this experiment (used in saved files)"
    )
    
    # Visualization
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization after training"
    )
    parser.add_argument(
        "--viz-only", type=str, default=None,
        help="Path to checkpoint file for visualization only (no training)"
    )
    
    # System
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Print banner
    print("=" * 80)
    print("VAE TRAINING ON CelebA DATASET")
    print("=" * 80)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    
    # Create config
    config = Config(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        dataset_limit=args.dataset_limit,
        beta=args.beta,
        likelihood=args.likelihood,
        num_workers=args.num_workers,
        device=device
    )
    
    print(f"\n{config}")
    
    # Generate experiment name
    if args.name:
        model_name = args.name
    else:
        model_name = f"vae_{config.likelihood}_beta{config.beta}"
    
    print(f"\n[INFO] Experiment name: {model_name}")
    
    # Determine normalization based on likelihood
    normalize_to_minus_one = config.likelihood == "gaussian"
    
    # Create dataloaders
    print(f"\n[INFO] Loading dataset from {config.dataset_dir}...")
    train_loader, test_loader = create_dataloaders(
        dataset_dir=config.dataset_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        limit=config.dataset_limit,
        train_split=config.train_split,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        normalize_to_minus_one=normalize_to_minus_one
    )
    
    # Visualization only mode
    if args.viz_only:
        print(f"\n[INFO] Loading model from {args.viz_only} for visualization...")
        from trainer import load_model
        model, checkpoint = load_model(Path(args.viz_only), device)
        history = checkpoint.get('history', None)
        
        run_visualizations(model, test_loader, device, config, model_name, history)
        return
    
    # Create model
    model = VAE(latent_dim=config.latent_dim, likelihood=config.likelihood).to(device)
    print(f"[MODEL] Parameters: {model.count_parameters():,}")
    
    # Create trainer and train
    trainer = VAETrainer(model, config, train_loader, test_loader)
    history = trainer.train(model_name=model_name)
    
    # Visualizations
    if not args.no_viz:
        run_visualizations(model, test_loader, device, config, model_name, history)
    
    # Print summary
    print_summary(history, model_name)


def run_visualizations(model, test_loader, device, config, model_name, history=None):
    """Run all visualizations."""
    print("\n" + "=" * 80)
    print("VISUALIZATIONS")
    print("=" * 80)
    
    # Training curves
    if history:
        plot_training_curves(
            history,
            f"{model_name} Training Curves",
            save_path=config.output_dir / f"{model_name}_curves.png"
        )
    
    # Reconstructions
    visualize_reconstructions(
        model, test_loader, device,
        num_images=5,
        title=f"{model_name} Reconstructions",
        save_path=config.output_dir / f"{model_name}_reconstructions.png"
    )
    
    # Latent interpolation
    latent_interpolation(
        model, test_loader, device,
        num_sequences=3,
        save_dir=config.output_dir,
        save_prefix=f"{model_name}_interpolation"
    )
    
    # Latent traversal
    latent_traversal(
        model, test_loader, device,
        num_dims=5,
        save_dir=config.output_dir,
        save_prefix=f"{model_name}_traversal"
    )
    
    # Generated samples
    generate_samples(
        model, device,
        num_samples=16,
        title=f"{model_name} Generated Samples",
        save_path=config.output_dir / f"{model_name}_samples.png"
    )


def print_summary(history, model_name):
    """Print training summary."""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    print(f"\n{model_name}:")
    print(f"  Final Train Loss: {history['train_total_loss'][-1]:.4f}")
    print(f"  Final Test Loss: {history['test_total_loss'][-1]:.4f}")
    print(f"  Best Test Loss: {min(history['test_total_loss']):.4f}")
    print(f"  Final Recon Loss: {history['test_recon_loss'][-1]:.4f}")
    print(f"  Final KL Loss: {history['test_kl_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
