import torch
import torch.utils.data
import torchvision.transforms as T
import os
import pickle
import cv2
import numpy as np
from typing import List, Dict

from model_utils.config_model_base import (get_model_instance_segmentation,
                                           NUM_CLASSES)
from model_utils import DEVICE, CHECKPOINT_DIR, MODEL_NAME_PREFIX
from data_utils.dataset_class import load_annotations

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME_PREFIX}_best.pth')
INPUT_ERROR_FILE = 'error_analysis_ids.pkl'
ANNOTATIONS_FILE = 'formatted_annotations.pkl'
CONFIDENCE_THRESHOLD = 0.75
OUTPUT_VIS_DIR = 'visualization_results'


def load_model(path: str) -> torch.nn.Module:
    """Loads the best model checkpoint."""
    model = get_model_instance_segmentation(NUM_CLASSES)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def draw_boxes(img_bgr: np.ndarray, boxes: np.ndarray, color,
               thickness: int, label: str):
    """Draws bounding boxes on an image """
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, thickness)


def visualize_detections(model: torch.nn.Module,
                         all_annotations: Dict[str, Dict],
                         error_ids: Dict[str, List[str]]):
    """Runs inference and visualizes ground truth vs predictions for selected IDs."""  # noqa E501

    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    transform = T.Compose([T.ToTensor()])

    for case_type, id_list in error_ids.items():
        print(f"Visualizing {case_type} cases...")
        for img_id in id_list:
            record = all_annotations.get(img_id)
            if not record:
                print(f"Warning: Image ID {img_id} not found in annotations dictionary.") # noqa E501
                continue

            img = cv2.imread(record['file_path'], cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"Warning: Could not read image at {record['file_path']}")  # noqa E501
                continue

            if img.dtype != np.uint8:
                img = (255 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())).astype(np.uint8) # noqa E501
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img_tensor = transform(img_bgr).to(DEVICE)

            with torch.no_grad():
                prediction = model([img_tensor])[0]

            high_conf_indices = prediction['scores'] > CONFIDENCE_THRESHOLD
            pred_boxes = prediction['boxes'][high_conf_indices].cpu().numpy()

            gt_boxes = np.array([ann['bbox'] for ann in record['annotations']]) if record['annotations'] else np.zeros((0, 4))  # noqa E501

            draw_boxes(img_bgr, gt_boxes, color=(0, 255, 0),
                       thickness=2, label="GT")

            draw_boxes(img_bgr, pred_boxes, color=(0, 0, 255),
                       thickness=1, label="PRED")

            legend_text_gt = "GT: Green"
            legend_text_pred = f"PRED: Red (Score>{CONFIDENCE_THRESHOLD})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            h, w, _ = img_bgr.shape

            cv2.putText(img_bgr, legend_text_gt, (10, h - 30), font,
                        font_scale, (0, 255, 0), font_thickness)
            cv2.putText(img_bgr, legend_text_pred, (10, h - 10), font,
                        font_scale, (0, 0, 255), font_thickness)

            output_path = os.path.join(OUTPUT_VIS_DIR,
                                       f"{case_type}_{img_id}.png")
            cv2.imwrite(output_path, img_bgr)

    print(f"\nVisualization complete. Images saved to {OUTPUT_VIS_DIR}")


if __name__ == '__main__':
    try:
        model = load_model(CHECKPOINT_PATH)
        all_annotations = load_annotations(ANNOTATIONS_FILE)

        with open(INPUT_ERROR_FILE, 'rb') as f:
            error_ids = pickle.load(f)

        visualize_detections(model, all_annotations, error_ids)

    except Exception as e:
        print(f"An error occurred during visualization: {e}")
