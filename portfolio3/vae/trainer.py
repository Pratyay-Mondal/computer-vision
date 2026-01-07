"""
Training utilities for VAE.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import VAE
from losses import compute_vae_loss
from config import Config


class VAETrainer:
    """Trainer class for VAE models."""
    
    def __init__(
        self,
        model: VAE,
        config: Config,
        train_loader: DataLoader,
        test_loader: DataLoader
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        self.history = {
            'train_total_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'test_total_loss': [],
            'test_recon_loss': [],
            'test_kl_loss': [],
            'epoch_times': []
        }
        
        self.best_test_loss = float('inf')
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(self.config.device)
            
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(data)
            
            loss, r_loss, k_loss = compute_vae_loss(
                recon, data, mu, logvar,
                beta=self.config.beta,
                likelihood=self.config.likelihood
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            recon_loss += r_loss.item()
            kl_loss += k_loss.item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item()/len(data):.4f}',
                    'Recon': f'{r_loss.item()/len(data):.4f}',
                    'KL': f'{k_loss.item()/len(data):.4f}'
                })
        
        num_samples = len(self.train_loader.dataset)
        return {
            'total_loss': total_loss / num_samples,
            'recon_loss': recon_loss / num_samples,
            'kl_loss': kl_loss / num_samples
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on test set."""
        self.model.eval()
        
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        
        for data in self.test_loader:
            data = data.to(self.config.device)
            recon, mu, logvar = self.model(data)
            
            loss, r_loss, k_loss = compute_vae_loss(
                recon, data, mu, logvar,
                beta=self.config.beta,
                likelihood=self.config.likelihood
            )
            
            total_loss += loss.item()
            recon_loss += r_loss.item()
            kl_loss += k_loss.item()
        
        num_samples = len(self.test_loader.dataset)
        return {
            'total_loss': total_loss / num_samples,
            'recon_loss': recon_loss / num_samples,
            'kl_loss': kl_loss / num_samples
        }
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'beta': self.config.beta,
                'likelihood': self.config.likelihood,
                'latent_dim': self.config.latent_dim
            }
        }
        
        if is_best:
            checkpoint['best_test_loss'] = self.best_test_loss
        
        torch.save(checkpoint, path)
    
    def train(self, model_name: str = "vae") -> dict:
        """
        Full training loop.
        
        Args:
            model_name: Name for saving checkpoints
            
        Returns:
            Training history
        """
        print(f"\n{'='*80}")
        print(f"TRAINING: {model_name.upper()}")
        print(f"Beta={self.config.beta}, Likelihood={self.config.likelihood}")
        print(f"{'='*80}\n")
        
        num_params = self.model.count_parameters()
        print(f"[MODEL] Total trainable parameters: {num_params:,}")
        
        for epoch in range(self.config.epochs):
            epoch_start = datetime.now()
            
            print(f"\n[Epoch {epoch+1}/{self.config.epochs}]")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            test_metrics = self.evaluate()
            
            # Record history
            self.history['train_total_loss'].append(train_metrics['total_loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['test_total_loss'].append(test_metrics['total_loss'])
            self.history['test_recon_loss'].append(test_metrics['recon_loss'])
            self.history['test_kl_loss'].append(test_metrics['kl_loss'])
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            self.history['epoch_times'].append(epoch_time)
            
            # Logging
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train - Total: {train_metrics['total_loss']:.4f} | "
                  f"Recon: {train_metrics['recon_loss']:.4f} | KL: {train_metrics['kl_loss']:.4f}")
            print(f"  Test  - Total: {test_metrics['total_loss']:.4f} | "
                  f"Recon: {test_metrics['recon_loss']:.4f} | KL: {test_metrics['kl_loss']:.4f}")
            
            # Save best model
            if test_metrics['total_loss'] < self.best_test_loss:
                self.best_test_loss = test_metrics['total_loss']
                best_path = self.config.output_dir / f"{model_name}_best.pt"
                self.save_checkpoint(best_path, is_best=True)
                print(f"  [CHECKPOINT] Saved best model (test_loss: {self.best_test_loss:.4f})")
        
        # Save final model
        final_path = self.config.output_dir / f"{model_name}_final.pt"
        self.save_checkpoint(final_path)
        
        # Save history as JSON
        history_path = self.config.output_dir / f"{model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n[COMPLETE] Training finished!")
        print(f"  Best test loss: {self.best_test_loss:.4f}")
        print(f"  Average epoch time: {np.mean(self.history['epoch_times']):.2f}s")
        print(f"  Models saved to: {self.config.output_dir}")
        
        return self.history


def load_model(
    checkpoint_path: Path,
    device: torch.device
) -> tuple[VAE, dict]:
    """
    Load a trained VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    latent_dim = config.get('latent_dim', 128)
    likelihood = config.get('likelihood', 'gaussian')
    
    model = VAE(latent_dim=latent_dim, likelihood=likelihood).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint
