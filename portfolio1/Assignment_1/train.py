import os
import pickle
import shutil
import time
from datetime import timedelta
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from data_utils import WaterfowlDataset, collate_fn, get_transform
from model_utils import (BASE_LR, BATCH_SIZE, CHECKPOINT_DIR, DEVICE, LOG_FREQ,
                         LR_GAMMA, LR_STEP_SIZE, MODEL_NAME_PREFIX, MOMENTUM,
                         NUM_CLASSES, NUM_EPOCHS, NUM_WORKERS, WEIGHT_DECAY,
                         get_model_instance_segmentation)


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str):
    """Saves model checkpoint and copies the 'best' one."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, f'{MODEL_NAME_PREFIX}_latest.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, f'{MODEL_NAME_PREFIX}_best.pth'))  # noqa: E501
    print(f"Checkpoint saved to {filepath}")
    if is_best:
        print(">>> New BEST model saved! <<<")


def train_one_epoch(model: torch.nn.Module, optimizer: optim.Optimizer,
                    data_loader: DataLoader, device: str, epoch: int):
    """Runs the training loop for one epoch."""
    model.train()
    start_time = time.time()
    total_loss = 0
    metric_logger: Dict[str, float] = {}

    print(f"\n--- Epoch {epoch} Start: LR = {optimizer.param_groups[0]['lr']:.6f} ---")  # noqa: E501

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        for loss_name, loss_value in loss_dict.items():
            loss_item = loss_value.item()
            metric_logger[loss_name] = metric_logger.get(loss_name, 0.0)
            + loss_item

        total_loss += losses.item()

        current_time = time.time()
        elapsed_time = current_time - start_time
        avg_iter_time = elapsed_time / (i + 1)
        loss_str = " | ".join([f"{k}: {v.item():.3f}" for k, v in loss_dict.items()])  # noqa: E501

        if (i + 1) % LOG_FREQ == 0 or i == len(data_loader) - 1:
            eta_seconds = avg_iter_time * (len(data_loader) - (i + 1))
            eta_str = str(timedelta(seconds=int(eta_seconds)))

            print(f"Epoch: [{epoch}][{i+1}/{len(data_loader)}] "
                  f"Loss: {losses.item():.4f} "
                  f"(Avg: {total_loss / (i+1):.4f}) | "
                  f"Individual: [{loss_str}] | "
                  f"Time: {avg_iter_time:.2f}s/iter | "
                  f"ETA: {eta_str}")

    avg_loss = total_loss / len(data_loader)
    avg_metric_logger = {k: v /
                         len(data_loader) for k, v in metric_logger.items()}

    return avg_loss, time.time() - start_time, avg_metric_logger


def get_validation_metrics(model: torch.nn.Module,
                           data_loader: DataLoader, device: str):
    """
    Returns: mAP score (float) and average validation losses (Dict).
    """
    model.eval()
    metric_calculator = MeanAveragePrecision(box_format='xyxy',
                                             iou_thresholds=None).to(device)

    total_val_loss = 0
    loss_metric_logger: Dict[str, float] = {}

    print("\n--- Starting Validation for mAP and Loss ---")

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]  # noqa: E501

            model.train()
            loss_dict = model(images, targets_on_device)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
            for loss_name, loss_value in loss_dict.items():
                loss_metric_logger[loss_name] = loss_metric_logger.get(loss_name, 0.0) + loss_value.item()  # noqa: E501
            model.eval()

            detections = model(images)
            metric_calculator.update(detections, targets_on_device)

    metrics = metric_calculator.compute()
    mAP_score = metrics['map'].item()

    avg_val_loss = total_val_loss / len(data_loader)
    avg_loss_metrics = {k: v / len(data_loader) for k, v in loss_metric_logger.items()}  # noqa: E501
    metric_calculator.reset()

    loss_str = " | ".join([f"Loss_{k}: {v:.4f}" for k, v in avg_loss_metrics.items()])  # noqa: E501

    print(f"Validation Loss (for log): {avg_val_loss:.4f} | mAP: {mAP_score:.4f}")  # noqa: E501
    print(f"Validation Individual Losses: [{loss_str}]")
    print("--- Validation Complete ---")

    return mAP_score, avg_loss_metrics


def main():
    print("--- Model Training ---")
    print(f"Device: {DEVICE}, Batch Size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

    print("Loading datasets...")
    train_dataset = WaterfowlDataset(split='train',
                                     transforms=get_transform(train=True))
    val_dataset = WaterfowlDataset(split='val',
                                   transforms=get_transform(train=False))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    print(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")  # noqa: E501

    model = get_model_instance_segmentation(NUM_CLASSES).to(DEVICE)
    print(f"Model initialized with {NUM_CLASSES} classes.")

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(params, lr=BASE_LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=LR_STEP_SIZE,
                                             gamma=LR_GAMMA)
    print("Optimizer: SGD | Scheduler: StepLR")

    best_val_mAP = 0.0
    total_start_time = time.time()

    history = {'train_loss': [], 'val_mAP': [], 'lr': []}
    TRAINING_HISTORY_FILE = 'training_history.pkl'

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, epoch_time, train_metrics = train_one_epoch(model,
                                                                optimizer,
                                                                train_loader,
                                                                DEVICE,
                                                                epoch)

        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
        next_lr = optimizer.param_groups[0]['lr']

        val_mAP, val_metrics = get_validation_metrics(model,
                                                      val_loader,
                                                      DEVICE)

        history['train_loss'].append(train_loss)
        history['val_mAP'].append(val_mAP)
        history['lr'].append(current_lr)

        with open(TRAINING_HISTORY_FILE, 'wb') as f:
            pickle.dump(history, f)

        is_best = val_mAP > best_val_mAP
        best_val_mAP = max(val_mAP, best_val_mAP)

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_mAP': best_val_mAP,
            'val_mAP': val_mAP,
            'train_loss': train_loss,
        }, is_best, CHECKPOINT_DIR)

        train_metrics_str = " | ".join([f"Train_{k}: {v:.4f}" for k, v in train_metrics.items()])  # noqa: E501
        val_metrics_str = " | ".join([f"Loss_{k}: {v:.4f}" for k, v in val_metrics.items()])  # noqa: E501

        total_elapsed_time = time.time() - total_start_time
        avg_epoch_time = total_elapsed_time / epoch
        eta_sec = avg_epoch_time * (NUM_EPOCHS - epoch)
        eta_str = str(timedelta(seconds=int(eta_sec)))

        print("\n" + "#" * 70)
        print(f"### EPOCH {epoch}/{NUM_EPOCHS} SUMMARY ###")
        print(f"LR Change: {current_lr:.6f} -> {next_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | **Val mAP: {val_mAP:.4f}** | **BEST Val mAP: {best_val_mAP:.4f}**")  # noqa: E501
        print(f"Time: {epoch_time:.0f}s | Total Time: {str(timedelta(seconds=int(total_elapsed_time)))} | ETA: {eta_str}")  # noqa: E501
        print(f"Detailed Train Losses: {train_metrics_str}")
        print(f"Detailed Val Losses:   {val_metrics_str}")
        print("#" * 70 + "\n")

    print("Training finished.")


if __name__ == '__main__':
    try:
        LOG_FREQ
    except NameError:
        LOG_FREQ = 5

    main()
