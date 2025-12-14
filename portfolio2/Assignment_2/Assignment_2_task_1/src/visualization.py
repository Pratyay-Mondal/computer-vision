import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.config import OUTPUT_DIR

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def visualize_kernels(model, title="Layer Weights"):
    model.eval()
    first_layer = model.features[0][0]
    weights = first_layer.weight.data.cpu().numpy()
    min_w = weights.min()
    max_w = weights.max()
    weights = (weights - min_w) / (max_w - min_w)
    num_filters = min(32, weights.shape[0])
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle(f"{title} - First 32 Filters", fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = weights[i, 0, :, :]
            ax.imshow(img, cmap='cividis')
            ax.axis('off')
            if np.isclose(img.sum(), 0):
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"viz_{title.replace(' ', '_')}.png"))
    plt.close()

def plot_confusion_matrix(model, dataloader, device, title="Confusion Matrix"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()

def plot_damage_recovery(results_log):
    df = pd.DataFrame(results_log)
    oneshot_df = df[df['experiment'].str.contains("OneShot")]
    if oneshot_df.empty: return

    oneshot_df = oneshot_df.sort_values('accuracy', ascending=False)
    labels = [e.replace("OneShot_", "") for e in oneshot_df['experiment']]
    accuracies = oneshot_df['accuracy'].values

    plt.figure(figsize=(9, 6))
    x = np.arange(len(labels))
    bars = plt.bar(x, accuracies, color='#4c72b0', width=0.6, edgecolor='black', alpha=0.9)
    plt.title("One-Shot Pruning Results", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Strategy", fontsize=12)
    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_oneshot_bar.png"), dpi=300)
    plt.close()

def plot_final_leaderboard(results_log, baseline_acc):
    df = pd.DataFrame(results_log)
    final_states = []
    for exp in df['experiment'].unique():
        last = df[df['experiment'] == exp].iloc[-1]
        final_states.append(last)

    final_df = pd.DataFrame(final_states).sort_values('accuracy', ascending=True)
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if 'OneShot' in x else '#1f77b4' for x in final_df['experiment']]
    bars = plt.barh(final_df['experiment'], final_df['accuracy'], color=colors, edgecolor='black', alpha=0.8)
    plt.axvline(baseline_acc, color='green', linestyle='--', linewidth=2, label='Baseline Acc')
    plt.title("Leaderboard: Accuracy & Compressed Size", fontsize=14, fontweight='bold')
    plt.xlabel("Accuracy (%)", fontsize=12)
    plt.xlim(0, 115)
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    for bar, size_mb in zip(bars, final_df['model_size']):
        width = bar.get_width()
        label_text = f"{width:.1f}%  ({size_mb:.2f} MB)"
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, label_text,
                 va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_leaderboard.png"), dpi=300)
    plt.close()

def plot_pareto_frontier(results_log, baseline_acc):
    df = pd.DataFrame(results_log)
    plt.figure(figsize=(12, 8))
    plt.axhline(y=baseline_acc, color='#2ca02c', linestyle='--', linewidth=2, 
                label=f'Baseline ({baseline_acc:.1f}%)', alpha=0.6)
    unique_exps = df['experiment'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, exp in enumerate(unique_exps):
        subset = df[df['experiment'] == exp].sort_values('sparsity')
        is_iterative = "Iterative" in exp
        linestyle = '-' if is_iterative else '--'
        marker = 'o' if is_iterative else 'D'
        color = colors[i % len(colors)]
        
        plt.plot(subset['sparsity'], subset['accuracy'], 
                 marker=marker, markersize=8, linestyle=linestyle, linewidth=2,
                 color=color, label=exp.replace("Iterative_", "Iter_").replace("OneShot_", "1Shot_"))

        if not subset.empty:
            last_point = subset.iloc[-1]
            plt.text(last_point['sparsity'], last_point['accuracy'] + 1, 
                     f"{last_point['accuracy']:.1f}%", 
                     fontsize=9, color=color, fontweight='bold', ha='center')

    plt.title("All Pruning Configurations", fontsize=15, fontweight='bold')
    plt.xlabel("Sparsity (Conv Layers) %", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(frameon=True, fontsize=10, loc='lower left')
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_pareto_frontier_all.png"), dpi=300)
    plt.close()

def plot_metrics_dashboard(results_log, experiment_name="Iterative_L1_90"):
    df = pd.DataFrame(results_log)
    
    subset = df[df['experiment'] == experiment_name].sort_values('sparsity')
    
    if subset.empty:
        print(f"No data found for {experiment_name}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    xlabel = "Sparsity (%)"
    
    ax1.plot(subset['sparsity'], subset['accuracy'], marker='o', color='#1f77b4', linewidth=2)
    ax1.set_ylabel("Accuracy (%)", fontweight='bold', color='#1f77b4')
    ax1.set_title(f"Metric Evolution: {experiment_name}", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    start_acc = subset.iloc[0]['accuracy']
    end_acc = subset.iloc[-1]['accuracy']
    ax1.annotate(f"Drop: {start_acc - end_acc:.1f}%", 
                 xy=(subset.iloc[-1]['sparsity'], end_acc),
                 xytext=(subset.iloc[-1]['sparsity'] - 20, end_acc + 2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.plot(subset['sparsity'], subset['inference_time'], marker='s', color='#ff7f0e', linewidth=2)
    ax2.set_ylabel("Inference Time (ms)", fontweight='bold', color='#ff7f0e')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(bottom=0) 
    ax2.set_ylim(top=subset['inference_time'].max() * 1.5) # Add headroom

    ax3.plot(subset['sparsity'], subset['model_size'], marker='d', color='#2ca02c', linewidth=2)
    ax3.set_ylabel("Model Size (MB)", fontweight='bold', color='#2ca02c')
    ax3.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_ylim(bottom=0)
    ax3.set_ylim(top=subset['model_size'].max() * 1.5) # Add headroom
    
    plt.tight_layout()
    plt.savefig("plot_metrics_dashboard.png", dpi=300)
    plt.show()
