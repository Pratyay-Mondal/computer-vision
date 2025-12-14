import json
import os

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from data_utils.data_augmentation import get_transform
from data_utils.dataset_class import WaterfowlDataset, collate_fn
from model_utils import CHECKPOINT_DIR, DEVICE, MODEL_NAME_PREFIX
from model_utils.config_model_base import (NUM_CLASSES,
                                           get_model_instance_segmentation)

BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR,
                                    f'{MODEL_NAME_PREFIX}_best.pth')
METRICS_OUTPUT_FILE = 'evaluation_metrics.json'


def main():
    print(f"Starting evaluation on device: {DEVICE}")

    model = get_model_instance_segmentation(NUM_CLASSES)

    if not os.path.exists(BEST_CHECKPOINT_PATH):
        print(f"ERROR: Best checkpoint not found at {BEST_CHECKPOINT_PATH}. Please train the model first.")  # noqa: E501
        return

    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    print("Loading test dataset...")
    test_dataset = WaterfowlDataset(split='test',
                                    transforms=get_transform(train=False))
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=None,
                                  max_detection_thresholds=None).to(DEVICE)

    print("Running inference on test set...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]  # noqa: E501
            predictions = model(images)

            preds_for_metric = []
            for p in predictions:
                preds_for_metric.append({
                    "boxes": p["boxes"],
                    "scores": p["scores"],
                    "labels": p["labels"].long()
                 })

            metric.update(preds_for_metric, targets)

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} batches...")

    results = metric.compute()
    results_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in results.items()}  # noqa: E501

    with open(METRICS_OUTPUT_FILE, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print("\n--- Evaluation Complete ---")
    print(f"Results saved to: {METRICS_OUTPUT_FILE}")
    print(f"mAP (IoU=0.50:0.95): {results_dict.get('map', 0.0):.4f}")
    print(f"mAP_50 (IoU=0.50): {results_dict.get('map_50', 0.0):.4f}")


if __name__ == '__main__':
    main()
