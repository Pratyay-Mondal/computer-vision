# VAE Training on CelebA Dataset

A PyTorch implementation of Variational Autoencoders (VAE) and β-VAE for the CelebA face dataset, supporting multiple likelihood distributions.

## Project Structure

```
vae/
├── dataset/
│   ├── img_align_celeba/    # CelebA images
│   ├── Anno/
│   │   └── list_attr_celeba.txt  # Attribute annotations
│   └── ...
├── outputs/                  # Training outputs (created automatically)
├── experiments/              # Standalone experiment scripts
│   ├── __init__.py
│   ├── utils.py              # Shared utilities
│   ├── attribute_direction.py  # Semantic direction discovery
│   ├── interpolation.py      # Latent interpolation
│   └── traversal.py          # Dimension traversal
├── config.py                 # Configuration dataclass
├── dataset.py                # CelebA dataset loader
├── model.py                  # VAE architecture
├── losses.py                 # Loss functions for different likelihoods
├── trainer.py                # Training utilities
├── visualization.py          # Visualization functions
├── train.py                  # Main training script
├── run_experiments.py        # Run multiple experiments
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Place the CelebA `img_align_celeba` folder in `./dataset/`:

```
dataset/
├── img_align_celeba/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── Anno/
├── Eval/
└── README.txt
```

## Usage

### Basic Training

Train a baseline VAE with Gaussian likelihood (default):

```bash
python train.py
```

### Likelihood Distributions

This implementation supports three likelihood distributions:

1. **Gaussian** (default): Uses MSE loss, output normalized to [-1, 1]
   ```bash
   python train.py --likelihood gaussian
   ```

2. **Continuous Bernoulli**: Proper distribution for continuous [0, 1] data
   ```bash
   python train.py --likelihood continuous_bernoulli
   ```

3. **Bernoulli**: Standard BCE loss (commonly used but technically incorrect for continuous data)
   ```bash
   python train.py --likelihood bernoulli
   ```

### β-VAE Training

Train β-VAE with higher KL weight for better disentanglement:

```bash
python train.py --beta 4.0 --name beta_vae_4
python train.py --beta 10.0 --name beta_vae_10
```

### Full Options

```bash
python train.py \
    --beta 1.0 \
    --likelihood gaussian \
    --epochs 10 \
    --batch-size 128 \
    --latent-dim 128 \
    --lr 1e-4 \
    --dataset-limit 200000 \
    --dataset-dir ./dataset/img_align_celeba \
    --output-dir ./outputs \
    --name my_experiment \
    --num-workers 4
```

### Run All Experiments

Train baseline and β-VAE variants with a single command:

```bash
# With Gaussian likelihood
python run_experiments.py --likelihood gaussian

# With Continuous Bernoulli
python run_experiments.py --likelihood continuous_bernoulli

# With standard Bernoulli
python run_experiments.py --likelihood bernoulli

# Custom beta values
python run_experiments.py --betas 1.0 2.0 4.0 8.0
```

### Visualization Only

Load a trained model and generate visualizations:

```bash
python train.py --viz-only ./outputs/vae_gaussian_beta1.0_best.pt
```

## Output Files

After training, you'll find in the output directory:

- `{name}_best.pt` - Best model checkpoint
- `{name}_final.pt` - Final model checkpoint
- `{name}_history.json` - Training history
- `{name}_curves.png` - Training curves plot
- `{name}_reconstructions.png` - Original vs reconstructed images
- `{name}_interpolation_seq*.png` - Latent space interpolations
- `{name}_traversal_dim*.png` - Latent dimension traversals
- `{name}_samples.png` - Samples from prior

## Model Architecture

The VAE uses a ResNet-style architecture:

**Encoder:**
- 4 ResBlockDown layers: 3→32→64→128→256 channels
- Spatial: 64×64 → 32×32 → 16×16 → 8×8 → 4×4
- Fully connected to μ and log σ² (latent dim: 128)

**Decoder:**
- Fully connected from latent space
- 4 ResBlockUp layers: 256→128→64→32→16 channels
- Final conv to 3 channels
- Output activation: Tanh (Gaussian) or Sigmoid (Bernoulli)

## Likelihood Comparison

| Likelihood | Output Range | Loss | Notes |
|------------|--------------|------|-------|
| Gaussian | [-1, 1] | MSE | Default, good general performance |
| Continuous Bernoulli | [0, 1] | -log_prob | Theoretically correct for continuous [0,1] data |
| Bernoulli | [0, 1] | BCE | Commonly used but incorrect for continuous data |

## Experiments

The `experiments/` folder contains standalone analysis scripts that don't modify the core training code.

### Attribute Direction

Find semantic directions in latent space based on CelebA attribute labels:

```bash
# List available attributes
python -m experiments.attribute_direction --ckpt outputs/vae_best.pt --list-attributes

# Compute and visualize "Smiling" direction
python -m experiments.attribute_direction --ckpt outputs/vae_best.pt --attribute Smiling

# Other attributes
python -m experiments.attribute_direction --ckpt outputs/vae_best.pt --attribute Eyeglasses
python -m experiments.attribute_direction --ckpt outputs/vae_best.pt --attribute Male --range 4.0
```

Available attributes include: Smiling, Eyeglasses, Male, Young, Bald, Bangs, Black_Hair, Blond_Hair, Heavy_Makeup, Wearing_Hat, etc.

### Latent Interpolation

Interpolate between image pairs in latent space:

```bash
# Basic interpolation
python -m experiments.interpolation --ckpt outputs/vae_best.pt

# More pairs and steps
python -m experiments.interpolation --ckpt outputs/vae_best.pt --pairs 10 --steps 15

# Spherical interpolation (slerp)
python -m experiments.interpolation --ckpt outputs/vae_best.pt --method spherical
```

### Latent Traversal

Traverse individual latent dimensions with variance-based selection:

```bash
# Traverse top-16 most variable dimensions
python -m experiments.traversal --ckpt outputs/vae_best.pt

# Traverse specific dimensions
python -m experiments.traversal --ckpt outputs/vae_best.pt --dims 0 5 10 15 20

# Without variance selection (first N dims)
python -m experiments.traversal --ckpt outputs/vae_best.pt --num-dims 10 --no-variance-selection

# Combined grid output
python -m experiments.traversal --ckpt outputs/vae_best.pt --combined-grid
```

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling, 2013
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) - Higgins et al., 2017
- [The Continuous Bernoulli: fixing a pervasive error in variational autoencoders](https://arxiv.org/abs/1907.06845) - Loaiza-Ganem & Cunningham, 2019
