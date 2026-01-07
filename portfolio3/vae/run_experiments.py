#!/usr/bin/env python3
"""
Run full experiments: baseline VAE and beta-VAE variants.

This script trains multiple VAE configurations and compares them.

Usage:
    python run_experiments.py
    python run_experiments.py --likelihood gaussian
    python run_experiments.py --likelihood continuous_bernoulli
    python run_experiments.py --likelihood bernoulli
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
    compare_models,
    generate_samples
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VAE experiments on CelebA dataset"
    )
    
    parser.add_argument(
        "--likelihood", type=str, default="gaussian",
        choices=["gaussian", "continuous_bernoulli", "bernoulli"],
        help="Likelihood distribution to use for all experiments"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="./dataset/img_align_celeba",
        help="Path to CelebA dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--dataset-limit", type=int, default=200000,
        help="Maximum number of images to use"
    )
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[1.0, 4.0, 10.0],
        help="Beta values to experiment with"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("VAE EXPERIMENTS ON CelebA DATASET")
    print(f"Likelihood: {args.likelihood}")
    print(f"Beta values: {args.betas}")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    
    output_dir = Path(args.output_dir) / f"experiments_{args.likelihood}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine normalization
    normalize_to_minus_one = args.likelihood == "gaussian"
    
    # Create dataloaders
    print(f"\n[INFO] Loading dataset...")
    train_loader, test_loader = create_dataloaders(
        dataset_dir=Path(args.dataset_dir),
        batch_size=128,
        limit=args.dataset_limit,
        normalize_to_minus_one=normalize_to_minus_one
    )
    
    # Store trained models and histories
    models = {}
    histories = {}
    
    # Train models with different beta values
    for beta in args.betas:
        model_name = f"vae_beta{beta}"
        
        print(f"\n{'='*80}")
        print(f"Training {model_name} ({args.likelihood} likelihood)")
        print("=" * 80)
        
        # Create config
        config = Config(
            output_dir=output_dir,
            epochs=args.epochs,
            beta=beta,
            likelihood=args.likelihood,
            device=device
        )
        
        # Create and train model
        model = VAE(latent_dim=config.latent_dim, likelihood=args.likelihood).to(device)
        trainer = VAETrainer(model, config, train_loader, test_loader)
        history = trainer.train(model_name=model_name)
        
        models[f"β={beta}"] = model
        histories[model_name] = history
        
        # Plot training curves
        plot_training_curves(
            history,
            f"VAE (β={beta}, {args.likelihood})",
            save_path=output_dir / f"{model_name}_curves.png"
        )
        
        # Visualizations
        visualize_reconstructions(
            model, test_loader, device,
            title=f"VAE (β={beta}) Reconstructions",
            save_path=output_dir / f"{model_name}_reconstructions.png"
        )
        
        latent_traversal(
            model, test_loader, device,
            num_dims=5,
            save_dir=output_dir,
            save_prefix=f"{model_name}_traversal"
        )
    
    # Compare all models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    compare_models(
        models,
        test_loader,
        device,
        num_images=5,
        save_path=output_dir / "model_comparison.png"
    )
    
    # Latent interpolation for baseline
    if 1.0 in args.betas:
        latent_interpolation(
            models["β=1.0"],
            test_loader,
            device,
            num_sequences=3,
            save_dir=output_dir,
            save_prefix="baseline_interpolation"
        )
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for name, history in histories.items():
        print(f"\n{name}:")
        print(f"  Final Train Loss: {history['train_total_loss'][-1]:.4f}")
        print(f"  Final Test Loss: {history['test_total_loss'][-1]:.4f}")
        print(f"  Best Test Loss: {min(history['test_total_loss']):.4f}")
        print(f"  Final Recon Loss: {history['test_recon_loss'][-1]:.4f}")
        print(f"  Final KL Loss: {history['test_kl_loss'][-1]:.4f}")
    
    print(f"\n[COMPLETE] All experiments finished!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
