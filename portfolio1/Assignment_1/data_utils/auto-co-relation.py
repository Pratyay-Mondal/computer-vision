import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def analyze_image_correlations(folder_path, resize_dim=(64, 64), uniqueness_threshold=0.6):
    """
    Analyzes sequential correlation and calculates the frame stride required for unique data.
    
    Args:
        folder_path: Path to images.
        resize_dim: Downsample size for speed.
        uniqueness_threshold: Correlation value below which images are considered 'unique' (0.6 is standard).
    """
    # 1. Load Images
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)])
    
    if len(files) < 2:
        print("Need at least 2 images to calculate correlation.")
        return

    print(f"Processing {len(files)} images...")
    
    images_flat = []
    valid_files = []

    for f in files:
        try:
            img_path = os.path.join(folder_path, f)
            # Convert to Grayscale (L)
            img = Image.open(img_path).convert('L').resize(resize_dim)
            arr = np.array(img).flatten()
            images_flat.append(arr)
            valid_files.append(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not images_flat:
        print("No valid images found.")
        return

    # Convert to matrix: (N_images, N_pixels)
    img_matrix = np.array(images_flat, dtype=np.float32)

    # 2. Compute Correlation Matrix (Vectorized)
    print("Computing similarity matrix...")
    means = img_matrix.mean(axis=1, keepdims=True)
    stds = img_matrix.std(axis=1, keepdims=True)
    stds[stds == 0] = 1e-6 # Avoid div by zero
    
    img_norm = (img_matrix - means) / stds
    sim_matrix = np.dot(img_norm, img_norm.T) / img_norm.shape[1]

    # 3. Calculate Uniqueness Decay (Lag Analysis)
    # We check lags 1, 2, 3... up to 50 (or half dataset size) to see when correlation drops
    max_lag = min(len(valid_files) // 2, 50) 
    lags = range(1, max_lag)
    avg_lag_corrs = []

    found_stride = False
    recommended_stride = -1

    for k in lags:
        # Extract the k-th diagonal (correlation between t and t+k)
        diag = sim_matrix.diagonal(offset=k)
        avg_corr = diag.mean()
        avg_lag_corrs.append(avg_corr)
        
        # Check if we hit the uniqueness threshold
        if not found_stride and avg_corr < uniqueness_threshold:
            recommended_stride = k
            found_stride = True

    # 4. Plotting
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Plot A: Lag-1 (Immediate neighbor)
    lag1_corrs = sim_matrix.diagonal(offset=1)
    ax[0].plot(range(1, len(valid_files)), lag1_corrs, color='tab:blue', alpha=0.7)
    ax[0].set_title('Step-by-Step Correlation')
    ax[0].set_xlabel('Frame Index')
    ax[0].set_ylabel('Correlation')
    ax[0].set_ylim(0, 1.05)
    ax[0].grid(True, alpha=0.3)

    # Plot B: Similarity Matrix
    im = ax[1].imshow(sim_matrix, cmap='inferno', vmin=0, vmax=1)
    ax[1].set_title('Similarity Matrix')
    ax[1].set_xlabel('Image Index')
    ax[1].set_ylabel('Image Index')
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    # Plot C: Correlation Decay (The "Uniqueness" Graph)
    ax[2].plot(lags, avg_lag_corrs, marker='o', markersize=4, color='tab:red')
    ax[2].axhline(uniqueness_threshold, color='green', linestyle='--', label=f'Uniqueness Threshold ({uniqueness_threshold})')
    
    if found_stride:
        ax[2].axvline(recommended_stride, color='green', linestyle=':', label=f'Unique at Lag {recommended_stride}')
        ax[2].plot(recommended_stride, avg_lag_corrs[recommended_stride-1], 'g*', markersize=15)
    
    ax[2].set_title('Correlation Decay')
    ax[2].set_xlabel('Lag (Frames apart)')
    ax[2].set_ylabel('Average Correlation')
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # 5. Interpretation & Recommendation
    print("-" * 50)
    print(f"RESULTS:")
    print(f"Current Lag-1 Correlation: {np.mean(lag1_corrs):.4f}")
    
    if found_stride:
        print(f"Images become unique (corr < {uniqueness_threshold}) after **{recommended_stride} frames**.")
        print(f"RECOMMENDATION: Train on every {recommended_stride}th image.")
    else:
        print(f"Correlation never dropped below {uniqueness_threshold} in the first {max_lag} frames.")
        print("RECOMMENDATION: Your data is extremely repetitive. Try a much larger stride or check for duplicates.")
    print("-" * 50)


# analyze_image_correlations('combined/')
