import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = "best_densenet121_chestxray.pth"
SAVE_XAI_GRADCAM = "xai_outputs/gradcam"
SAVE_XAI_LIME = "xai_outputs/lime"

MEAN = 0.482
STD = 0.223
TARGET_SIZE = 224   # final size: 224 x 224
NUM_XAI_IMAGES = 15


# =============================================================================
# Optional — Print GPU Memory
# =============================================================================

def print_gpu_memory():
    if not torch.cuda.is_available():
        print("\n[INFO] CUDA not available.")
        return
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print("\n=== GPU MEMORY ===")
    print(f"Total:      {total/1024**2:.1f} MB")
    print(f"Free:       {free/1024**2:.1f} MB")
    print(f"Allocated:  {allocated/1024**2:.1f} MB")
    print(f"Reserved:   {reserved/1024**2:.1f} MB")
    print("==================\n")


# =============================================================================
# Fix DenseNet in-place ReLUs (CRITICAL for Grad-CAM)
# =============================================================================

def fix_densenet_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def patch_densenet_forward(model):
    """
    DenseNet uses F.relu(..., inplace=True) inside forward().
    We patch the forward method to remove that last in-place op.
    """
    def new_forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False)    # <-- FIX
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    import types
    model.forward = types.MethodType(new_forward, model)


# =============================================================================
# Hard Resize to 224x224 (NO aspect ratio preservation, NO padding)
# =============================================================================

resize_to_224 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    # grayscale -> 3 channels (DenseNet expects 3-channel input)
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[MEAN] * 3, std=[STD] * 3),
])

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[MEAN] * 3, std=[STD] * 3),
])

val_test_transform = resize_to_224


# =============================================================================
# Dataset Paths
# =============================================================================

DATA_ROOT = "data/chest_xray/chest_xray/train"
TEST_ROOT = "data/chest_xray/chest_xray/test"


# =============================================================================
# Load datasets
# =============================================================================

full_train_dataset = datasets.ImageFolder(DATA_ROOT, transform=train_transform)
class_names = full_train_dataset.classes  # typically ['NORMAL', 'PNEUMONIA']

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_test_transform

test_dataset = datasets.ImageFolder(TEST_ROOT, transform=val_test_transform)


# =============================================================================
# Weighted sampling (handle imbalance)
# =============================================================================

all_targets = [label for _, label in full_train_dataset.samples]
train_targets = [all_targets[i] for i in train_dataset.indices]

class_counts = np.bincount(train_targets)
class_weights = 1.0 / class_counts

train_weights = [class_weights[t] for t in train_targets]
sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)


# =============================================================================
# Model Setup
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
fix_densenet_inplace_relu(model)
patch_densenet_forward(model)

num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.BatchNorm1d(num_features),
    nn.Dropout(0.4),
    nn.Linear(num_features, 2),
)

model = model.to(device)


# =============================================================================
# Load or Train
# =============================================================================

if os.path.exists(MODEL_PATH):
    print("\n✔ Pretrained model found — loading checkpoint...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    skip_training = True
else:
    print("\n✘ No pretrained model found — training from scratch...")
    skip_training = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()


# =============================================================================
# Training
# =============================================================================

if not skip_training:
    EPOCHS = 600
    patience = 30
    no_improve = 0
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("--------------------------------")

        # ---- Train ----
        model.train()
        t_loss = 0
        t_corr = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item() * x.size(0)
            t_corr += torch.sum(out.argmax(1) == y)

        t_loss /= len(train_dataset)
        t_acc = t_corr.double() / len(train_dataset)
        print(f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}")

        # ---- Validation ----
        model.eval()
        v_loss = 0
        v_corr = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = criterion(out, y)

                v_loss += loss.item() * x.size(0)
                v_corr += torch.sum(out.argmax(1) == y)

        v_loss /= len(val_dataset)
        v_acc = v_corr.double() / len(val_dataset)
        print(f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

        # ---- Early stopping ----
        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODEL_PATH)
            print("Model improved — checkpoint saved.")
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs.")

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_weights)
    print("\nTraining complete. Best validation loss:", best_loss)

else:
    print("\nTraining skipped — using pretrained model.")


# =============================================================================
# Denormalize for visualization
# =============================================================================

def denormalize(img):
    """
    img: tensor (3, H, W) normalized
    returns: tensor (3, H, W) in [0,1]
    """
    mean = torch.tensor([MEAN] * 3).view(3, 1, 1)
    std = torch.tensor([STD] * 3).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


# =============================================================================
# Collect fixed XAI samples (same 15 images for Grad-CAM and LIME)
# =============================================================================

def collect_fixed_xai_samples(loader, num_samples=15):
    images = []
    labels = []

    for x, y in loader:
        for i in range(x.size(0)):
            images.append(x[i].unsqueeze(0))  # keep batch dimension
            labels.append(y[i].item())
            if len(images) >= num_samples:
                imgs = torch.cat(images, dim=0)
                return imgs, labels

    # Fallback if dataset smaller than num_samples
    if len(images) > 0:
        imgs = torch.cat(images, dim=0)
        return imgs, labels
    else:
        raise RuntimeError("No images found in loader for XAI sampling.")


# =============================================================================
# Grad-CAM
# =============================================================================

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.activations = None
        self.gradients = None

        self.fwd = layer.register_forward_hook(self.fwd_hook)
        self.bwd = layer.register_full_backward_hook(self.bwd_hook)

    def fwd_hook(self, m, inp, out):
        self.activations = out.detach()

    def bwd_hook(self, m, gin, gout):
        self.gradients = gout[0].detach()

    def generate(self, x):
        self.model.zero_grad()
        out = self.model(x)
        c = out.argmax(1).item()
        out[0, c].backward()

        A = self.activations[0]
        G = self.gradients[0]

        w = G.mean(dim=(1, 2))
        cam = torch.zeros_like(A[0])

        for i, wi in enumerate(w):
            cam += wi * A[i]

        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return cam.squeeze().cpu().numpy()


def generate_gradcam_fixed(model, images, labels, device,
                           output_dir=SAVE_XAI_GRADCAM):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    camlayer = model.features[-1]
    gc = GradCAM(model, camlayer)

    for idx in range(len(images)):
        img_norm = images[idx].to(device).unsqueeze(0)  # (1,3,H,W)
        true_label = labels[idx]

        # Prediction
        with torch.no_grad():
            logits = model(img_norm)
            pred_label = logits.argmax(1).item()

        # Grad-CAM heatmap (needs gradients)
        heat = gc.generate(img_norm)

        # Denormalize for visualization
        base = denormalize(images[idx].cpu()).permute(1, 2, 0).numpy()

        correctness = "(CORRECT)" if pred_label == true_label else "(WRONG)"
        title = f"True: {class_names[true_label]} | Pred: {class_names[pred_label]} {correctness}"

        plt.figure(figsize=(4, 4))
        plt.imshow(base, alpha=0.7)
        plt.imshow(heat, cmap="jet", alpha=0.4)
        plt.title(title)
        plt.axis("off")
        plt.savefig(f"{output_dir}/gradcam_{idx:02d}.png",
                    bbox_inches="tight", pad_inches=0)
        plt.close()


# =============================================================================
# LIME Image Explainer (Superpixels) using same 15 images
# =============================================================================

def lime_classifier_fn(images_np):
    """
    images_np: (N, H, W, 3), uint8 or float in [0,255] or [0,1]
    Returns: (N, num_classes) probabilities.
    """
    model.eval()

    imgs = images_np.astype(np.float32)
    if imgs.max() > 1.0:
        imgs = imgs / 255.0  # scale to [0,1]

    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # (N,3,H,W)

    # Convert to grayscale-like (since training used grayscale -> repeat)
    gray = imgs_t.mean(dim=1, keepdim=True)   # (N,1,H,W)
    imgs_3 = gray.repeat(1, 3, 1, 1)          # (N,3,H,W)

    mean = torch.tensor([MEAN] * 3).view(1, 3, 1, 1)
    std = torch.tensor([STD] * 3).view(1, 3, 1, 1)
    imgs_3 = (imgs_3 - mean) / std

    imgs_3 = imgs_3.to(device)

    with torch.no_grad():
        logits = model(imgs_3)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


def generate_lime_fixed(model, images, labels, device,
                        output_dir=SAVE_XAI_LIME,
                        num_samples=1000):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    explainer = lime_image.LimeImageExplainer()

    for idx in range(len(images)):
        # Convert normalized tensor -> uint8 image for LIME
        img_denorm = denormalize(images[idx].cpu())           # (3,H,W) in [0,1]
        img_np = img_denorm.permute(1, 2, 0).numpy()          # (H,W,3)
        img_uint8 = (img_np * 255).astype(np.uint8)

        # Get prediction for title
        preds = lime_classifier_fn(img_uint8[np.newaxis, ...])
        pred_label = int(np.argmax(preds))

        explanation = explainer.explain_instance(
            img_uint8,
            classifier_fn=lime_classifier_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )

        # Explanation for predicted class
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label=top_label,
            positive_only=True,
            num_features=7,
            hide_rest=False
        )

        lime_vis = mark_boundaries(temp / 255.0, mask)

        title = f"LIME — True: {class_names[labels[idx]]} | Pred: {class_names[pred_label]}"

        fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
        axes[0].imshow(img_uint8, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(lime_vis)
        axes[1].set_title(title)
        axes[1].axis("off")

        save_path = os.path.join(output_dir, f"lime_{idx:02d}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


# =============================================================================
# Run XAI
# =============================================================================

print("\n=== Collecting fixed XAI samples ===")
xai_images, xai_labels = collect_fixed_xai_samples(test_loader, num_samples=NUM_XAI_IMAGES)

print("\n=== Running Grad-CAM on fixed samples ===")
generate_gradcam_fixed(model, xai_images, xai_labels, device)

print("\n=== Running LIME on same fixed samples ===")
generate_lime_fixed(model, xai_images, xai_labels, device)

print("\nAll done! Grad-CAM and LIME outputs saved under 'xai_outputs/'.")
                   
