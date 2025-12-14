import torch
import torch.utils.data
from torch.utils.data import DataLoader
import os
import pickle
import random
from typing import List, Dict

from torchvision.ops import box_iou

from data_utils.dataset_class import WaterfowlDataset, collate_fn
from model_utils.config_model_base import (get_model_instance_segmentation,
                                           NUM_CLASSES)
from model_utils import DEVICE, CHECKPOINT_DIR, MODEL_NAME_PREFIX
from data_utils.data_augmentation import get_transform

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME_PREFIX}_best.pth')
OUTPUT_ERROR_FILE = 'error_analysis_ids.pkl'
CONFIDENCE_THRESHOLD = 0.75
MIN_IOU_THRESHOLD = 0.50


def analyze_batch_rigorous(predictions: List[Dict], targets: List[Dict],
                           image_ids: List[str], conf_threshold: float,
                           iou_threshold: float = MIN_IOU_THRESHOLD):
    """
    Analyzes a batch to categorize images
    into TP, FN, and FP based on IoU overlap.
    A simple heuristic (80% accuracy/miss rate)
    is used for visualization selection.
    """

    results = {'TP': set(), 'FN': set(), 'FP': set()}

    for pred, target, img_id in zip(predictions, targets, image_ids):
        high_conf_preds = pred['scores'] > conf_threshold
        pred_boxes = pred['boxes'][high_conf_preds].cpu()
        gt_boxes = target['boxes'].cpu()

        num_gt = gt_boxes.shape[0]
        num_pred = pred_boxes.shape[0]

        if num_pred > 0 and num_gt > 0:
            iou_matrix = box_iou(pred_boxes, gt_boxes)
            max_iou_per_pred, _ = iou_matrix.max(dim=1)
            max_iou_per_gt, _ = iou_matrix.max(dim=0)

            num_tp = (max_iou_per_pred >= iou_threshold).sum().item()
            num_fn = (max_iou_per_gt < iou_threshold).sum().item()
            num_fp = num_pred - num_tp

            if num_tp >= num_gt * 0.8 and num_tp > 0:
                results['TP'].add(img_id)
            elif num_fn >= num_gt * 0.8 and num_gt > 0:
                results['FN'].add(img_id)

            elif num_fp >= num_pred * 0.8 and num_pred > 0:
                results['FP'].add(img_id)

        elif num_gt == 0 and num_pred > 0:
            results['FP'].add(img_id)

    return {k: list(v) for k, v in results.items()}


def main():
    print(f"Starting error analysis on device: {DEVICE}")
    try:
        model = get_model_instance_segmentation(NUM_CLASSES)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"ERROR loading model: {e}. Ensure functions/constants are defined.") # noqa E501
        return

    test_dataset = WaterfowlDataset(split='test',
                                    transforms=get_transform(train=False))
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    all_error_ids = {'TP': set(), 'FN': set(), 'FP': set()}

    print("Running inference and error categorization...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]  # noqa E501
            predictions = model(images)

            batch_indices = [t['image_id'].item() for t in targets]
            batch_img_ids = [test_dataset.records[idx]['image_id'] for idx in batch_indices]  # noqa E501

            batch_errors = analyze_batch_rigorous(predictions,
                                                  targets,
                                                  batch_img_ids,
                                                  CONFIDENCE_THRESHOLD,
                                                  MIN_IOU_THRESHOLD)

            for k, v in batch_errors.items():
                all_error_ids[k].update(v)

    final_error_ids = {
        'TP': random.sample(list(all_error_ids['TP']), min(len(all_error_ids['TP']), 3)),  # noqa E501
        'FN': random.sample(list(all_error_ids['FN']), min(len(all_error_ids['FN']), 3)),  # noqa E501
        'FP': random.sample(list(all_error_ids['FP']), min(len(all_error_ids['FP']), 3)),  # noqa E501
    }

    with open(OUTPUT_ERROR_FILE, 'wb') as f:
        pickle.dump(final_error_ids, f)

    print("\n--- Error Analysis Complete ---")
    for k, v in final_error_ids.items():
        print(f"Selected {len(v)} IDs for {k} Visualization: {v}")


if __name__ == '__main__':
    main()
