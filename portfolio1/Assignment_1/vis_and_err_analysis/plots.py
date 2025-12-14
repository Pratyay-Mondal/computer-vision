import matplotlib.pyplot as plt
import pickle
import json
import os

TRAINING_HISTORY_FILE = 'training_history.pkl'
METRICS_OUTPUT_FILE = 'evaluation_metrics.json'
OUTPUT_PLOTS_DIR = 'plots'


def load_history(file_path: str) -> dict:
    """Loads the training history (losses, mAP, LR) from the pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training history not found: {file_path}. Run model_trainer.py first.")  # noqa E501
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
        if 'val_mAP' not in history:
            raise KeyError("History file missing expected key 'val_mAP'. Check trainer script.")  # noqa E501


def load_metrics(file_path: str) -> dict:
    """Loads the final evaluation metrics from the JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation metrics not found: {file_path}. Run evaluation_metrics.py first.")  # noqa E501
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_performance_curves(history: dict, output_dir: str):
    """Plots the Training Loss and Validation mAP over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, history['train_loss'], color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation mAP', color=color)
    ax2.plot(epochs, history['val_mAP'], color=color, linestyle='--', label='Validation mAP') # noqa E501
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0)

    plt.title('Training Loss and Validation mAP')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.savefig(os.path.join(output_dir, 'performance_curves.png'))
    plt.close(fig)
    print("Saved performance curves plot (Loss vs. mAP).")


def plot_learning_rate(history: dict, output_dir: str):
    """Plots the learning rate schedule over epochs."""
    epochs = range(1, len(history['lr']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['lr'], 'g', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
    plt.close()
    print("Saved learning rate plot.")


def print_final_metrics(metrics: dict):
    """Prints a clean summary of the final evaluation metrics."""
    print("\n--- Final Evaluation Metrics Summary (Test Set) ---")

    mAP = metrics.get('map')
    mAP_50 = metrics.get('map_50')
    mAP_75 = metrics.get('map_75')
    mar_100 = metrics.get('mar_100')

    print(f"Standard mAP (IoU=0.50:0.95): {mAP:.4f}" if mAP is not None else "mAP: N/A")  # noqa E501
    print(f"mAP@50 (IoU=0.50, Localization): {mAP_50:.4f}" if mAP_50 is not None else "mAP@50: N/A")   # noqa E501
    print(f"mAP@75 (IoU=0.75, Strict):    {mAP_75:.4f}" if mAP_75 is not None else "mAP@75: N/A")  # noqa E501
    print(f"Avg Recall (Max Dets=100):   {mar_100:.4f}" if mar_100 is not None else "MAR@100: N/A")  # noqa E501
    print("-" * 50)


if __name__ == '__main__':
    try:
        os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

        history = load_history(TRAINING_HISTORY_FILE)
        metrics = load_metrics(METRICS_OUTPUT_FILE)

        plot_performance_curves(history, OUTPUT_PLOTS_DIR)
        plot_learning_rate(history, OUTPUT_PLOTS_DIR)

        print_final_metrics(metrics)

        print("\nAll necessary plots and metric summary generated.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except KeyError as e:
        print(f"ERROR: Incorrect data structure in history file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
