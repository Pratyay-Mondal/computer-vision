import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from src.config import experiments_config, MODEL_PATH, OUTPUT_DIR
from src.data import get_dataloaders
from src.utils import ensure_dir, save_json, save_text
from src.engine import train_one_epoch, evaluate_accuracy, run_experiment
from src.visualization import (
    plot_pareto_frontier, plot_damage_recovery, plot_final_leaderboard,
    plot_confusion_matrix, visualize_kernels
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    ensure_dir(OUTPUT_DIR)

    print("Preparing Data...")
    trainloader, testloader, testset = get_dataloaders()

    if os.path.exists(MODEL_PATH):
        print(f"Found saved model at {MODEL_PATH}. Loading...")
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
    else:
        print("Saved model not found. Starting training...")
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
        model = model.to(device)

        print("Training Baseline (10 Epochs)...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        baseline_epochs = 10
        for epoch in range(baseline_epochs):
            loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{baseline_epochs} | Loss: {loss:.4f}")
        
        torch.save(model.state_dict(), MODEL_PATH)

    model.eval()
    base_acc = evaluate_accuracy(model, testloader, device)
    print(f"\nBASELINE ACCURACY: {base_acc:.2f}%")
    plot_confusion_matrix(model, testloader, device, title="Baseline (0% Pruned)")
    visualize_kernels(model, "Baseline")

    results_log = []
    trained_models = {}

    for config in experiments_config:
        trained_models[config['name']] = run_experiment(
            model, config, trainloader, testloader, testset, device, results_log
        )

    print("\nGenerating Plots...")
    plot_pareto_frontier(results_log, base_acc)
    plot_damage_recovery(results_log)
    plot_final_leaderboard(results_log, base_acc)
    plot_metrics_dashboard(results_log, "Iterative_L1_90")

    print("Generating Pruned Confusion Matrix...")
    plot_confusion_matrix(trained_models['Iterative_L1_90'], testloader, device, title="Iterative L1 (90% Pruned)")

    visualize_kernels(trained_models['OneShot_Random_90'], "Pruned_90_Percent_OneShot_Random")
    visualize_kernels(trained_models['Iterative_Structured_90'], "Pruned_90_Percent_Iterative_Structured")
    visualize_kernels(trained_models['Iterative_L1_90'], "Pruned_90_Percent_Iterative_L1")

    print("\nSaving data for offline analysis...")
    df_results = pd.DataFrame(results_log)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "pruning_experiment_stats.csv"), index=False)
    save_json(experiments_config, os.path.join(OUTPUT_DIR, "experiment_config.json"))
    save_text(base_acc, os.path.join(OUTPUT_DIR, "baseline_acc.txt"))

if __name__ == "__main__":
    main()
