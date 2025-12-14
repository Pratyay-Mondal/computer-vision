import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def check_sparsity(model):
    sum_zeros = 0
    sum_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            sum_zeros += torch.sum(weight == 0).item()
            sum_params += weight.nelement()
    return 0.0 if sum_params == 0 else 100 * sum_zeros / sum_params

def remove_pruning_reparam(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass

def apply_pruning_step(model, method, amount):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    if method == "unstructured_l1":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    elif method == "unstructured_random":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=amount,
        )
    elif method == "structured_l2":
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
