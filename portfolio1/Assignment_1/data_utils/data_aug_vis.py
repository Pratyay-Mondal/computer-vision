import pickle
import random
from typing import Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_augmentation import get_transform

INPUT_FILE = '../formatted_annotations.pkl'
TRAIN_IDS_FILE = '../split_dataset/train_ids.txt'


def load_data_sample(image_id: str, sample_idx: int):
    try:
        with open(INPUT_FILE, 'rb') as f:
            all_metadata = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Annotation file not found at: {INPUT_FILE}")

    record = next((r for r in all_metadata if r['image_id'] == image_id), None)
    if record is None:
        raise ValueError(f"Metadata for ID {image_id} not found in {INPUT_FILE}.")  # noqa: E501

    image_path = "../" + record['file_path']
    image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image_np is None:
        raise FileNotFoundError(f"Image not found at {image_path}. Check path/access.")  # noqa: E501

    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=2)

    boxes_list = [ann['bbox'] for ann in record.get('annotations', [])]
    labels_list = [ann['category_id'] for ann in record.get('annotations', [])]

    target = {}
    target["boxes"] = torch.as_tensor(boxes_list, dtype=torch.float32)
    target["labels"] = torch.as_tensor(labels_list, dtype=torch.int64)
    target["image_id"] = torch.tensor([sample_idx])

    return image_np, target


def get_random_train_id(file_path: str) -> Tuple[str, int]:
    with open(file_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    idx = random.randrange(len(ids))
    return ids[idx], idx


def visualize_augmentation(transform_pipeline: A.Compose, num_samples: int = 4):  # noqa: E501
    print(f"Loading data from: {INPUT_FILE}")
    print(f"Using training IDs from: {TRAIN_IDS_FILE}")

    try:
        image_id, sample_idx = get_random_train_id(TRAIN_IDS_FILE)

        original_image_np, original_target = load_data_sample(image_id, sample_idx)  # noqa: E501
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"\n--- Visualizing Augmentations for Image ID: **{image_id}** ---")
    print(f"Original Shape: {original_image_np.shape}")
    print(f"Original BBoxes: {original_target['boxes'].shape[0]}")

    _, axes = plt.subplots(1, num_samples + 1, figsize=(4 * (num_samples + 1), 5))  # noqa: E501

    if num_samples == 0:
        axes = [axes]

    def draw_boxes(ax, image, boxes, title):
        if image.ndim == 3 and image.shape[2] == 1:
            display_image = np.tile(image, (1, 1, 3))
        else:
            display_image = image

        ax.imshow(display_image)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        for box in boxes:
            xmin, ymin, xmax, ymax = [int(b) for b in box]
            width = xmax - xmin
            height = ymax - ymin

            rect = plt.Rectangle((xmin, ymin), width, height,
                                 fill=False, color='red', linewidth=2)
            ax.add_patch(rect)

    original_boxes_pascal = original_target['boxes'].cpu().numpy()
    draw_boxes(axes[0], original_image_np, original_boxes_pascal, "Original")

    for i in range(num_samples):
        current_target = {k: v.clone() for k, v in original_target.items() if isinstance(v, torch.Tensor)} # noqa

        image_tensor, augmented_target = transform_pipeline(original_image_np,
                                                            current_target)

        augmented_image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # C, H, W -> H, W, C # noqa: E501

        augmented_image_np = (augmented_image_np * 0.5 + 0.5) * 255.0
        augmented_image_np = augmented_image_np.astype(np.uint8)

        augmented_boxes_pascal = augmented_target['boxes'].cpu().numpy()

        if augmented_boxes_pascal.size > 0:
            pass
        else:
            print(f"Sample {i+1} has **0 BBoxes** after augmentation.")

        draw_boxes(axes[i + 1], augmented_image_np, augmented_boxes_pascal, f"Augmented {i+1}")  # noqa: E501

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_transform = get_transform(train=True)
    visualize_augmentation(train_transform, num_samples=4)
