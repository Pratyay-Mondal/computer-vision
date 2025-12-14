import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from src.utils import get_model_size_mb
from src.pruning import check_sparsity, apply_pruning_step, remove_pruning_reparam

def measure_inference_time(model, device, dataset, num_samples=50):
    model.eval()
    image, _ = dataset[0]
    input_tensor = image.unsqueeze(0).to(device)
    for _ in range(5): _ = model(input_tensor)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(input_tensor)
            if device.type == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    return ((end_time - start_time) / num_samples) * 1000

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(trainloader)

def run_experiment(model, config, trainloader, testloader, testset, device, results_log):
    exp_name = config['name']
    print(f"\n=== STARTING: {exp_name} ===")
    curr_model = copy.deepcopy(model)
    optimizer = optim.SGD(curr_model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    target_sparsity = config.get('target_sparsity', 0.5)
    steps = config.get('steps', 1)
    epochs_per_step = config.get('epochs', 1)

    if steps > 1:
        prune_amount_per_step = 1 - (1 - target_sparsity)**(1/steps)
    else:
        prune_amount_per_step = target_sparsity

    for step in range(1, steps + 1):
        apply_pruning_step(curr_model, config['method'], prune_amount_per_step)
        print(f"--- Step {step}/{steps} [Pruning {prune_amount_per_step:.2%}] ---")
        for _ in range(epochs_per_step):
            train_one_epoch(curr_model, trainloader, criterion, optimizer, device)
        
        sparsity = check_sparsity(curr_model)
        acc = evaluate_accuracy(curr_model, testloader, device)
        inf_time = measure_inference_time(curr_model, device, testset)
        model_size = get_model_size_mb(curr_model)
        non_zero_params = sum(torch.sum(p != 0).item() for p in curr_model.parameters())
        total_params = sum(p.nelement() for p in curr_model.parameters())

        print(f"   -> Non-Zero Params: {non_zero_params}/{total_params} ({100 * non_zero_params/total_params:.2f}% active)")
        print(f"   -> Sparsity (Conv Layers): {sparsity:.1f}% | Acc: {acc:.1f}% | Time: {inf_time:.2f}ms")

        results_log.append({
            "experiment": exp_name,
            "stage": f"Step {step}" if steps > 1 else "One-Shot Result",
            "sparsity": sparsity,
            "accuracy": acc,
            "inference_time": inf_time,
            "model_size": model_size,
            "non_zero_params": non_zero_params,
            "total_params": total_params
        })

    remove_pruning_reparam(curr_model)
    return curr_model
