import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "best_densenet121_chestxray.pth"
TEST_DIR = "data/chest_xray/chest_xray/test"

BATCH_SIZE = 16
MEAN = 0.482
STD = 0.223

# Where to save all figures / reports
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Test Transform (MUST MATCH TRAINING)
# ============================================================

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),                 # HARD resize
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1 channel -> 3 channels
    transforms.Normalize(mean=[MEAN] * 3, std=[STD] * 3)
])


# ============================================================
# Load Test Dataset
# ============================================================

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
class_names = test_dataset.classes
print("Classes:", class_names)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# Load DenseNet Model (MUST MATCH TRAINING ARCHITECTURE)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
num_features = model.classifier.in_features

# ---- IMPORTANT: this classifier MUST match your training script ----
model.classifier = nn.Sequential(
    nn.BatchNorm1d(num_features),
    nn.Dropout(0.4),
    nn.Linear(num_features, 2)
)
# --------------------------------------------------------------------

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("\n✔ Model loaded successfully.\n")


# ============================================================
# Helper: Plot & Save Confusion Matrix
# ============================================================

def save_confusion_matrix(cm, class_names, out_path):
    """
    Save confusion matrix as a heatmap figure.
    cm: 2x2 numpy array
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )

    # Rotate tick labels and set alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Annotate each cell with its count
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs = model(x)               # logits
            preds = outputs.argmax(1)        # hard predictions

            # probability of PNEUMONIA (assumed class index 1)
            soft = torch.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(soft.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ===============================
    # Confusion Matrix
    # ===============================
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("=== CONFUSION MATRIX ===")
    print(cm)
    print(f"\nTN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")

    # Save CM figure
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    save_confusion_matrix(cm, class_names, cm_path)
    print(f"\nConfusion matrix figure saved to: {cm_path}")

    # ===============================
    # Classification Report
    # ===============================
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )

    print("\n=== CLASSIFICATION REPORT ===")
    print(report_str)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report_str)
    print(f"Classification report saved to: {report_path}")

    # ===============================
    # Manual Metrics
    # ===============================
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0 else 0.0
    )

    print("\n=== METRICS ===")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"F1 Score:     {f1:.4f}")

    # Save metrics summary
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        f.write("=== METRICS SUMMARY ===\n")
        f.write(f"Accuracy:     {accuracy:.4f}\n")
        f.write(f"Sensitivity:  {sensitivity:.4f}\n")
        f.write(f"Specificity:  {specificity:.4f}\n")
        f.write(f"Precision:    {precision:.4f}\n")
        f.write(f"F1 Score:     {f1:.4f}\n")
    print(f"Metrics summary saved to: {metrics_path}")

    # ===============================
    # ROC + AUC
    # ===============================
    auc = roc_auc_score(all_labels, all_probs)
    print(f"AUC Score:    {auc:.4f}")

    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve figure saved to: {roc_path}")

    return cm


# ============================================================
# Run Evaluation
# ============================================================

if __name__ == "__main__":
    evaluate_model(model, test_loader, device)
