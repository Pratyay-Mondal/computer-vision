import os
import pickle
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
from torch.utils.data import Dataset

INPUT_FILE = 'formatted_annotations.pkl'
TRAIN_IDS_FILE = 'split_dataset/train_ids.txt'
VAL_IDS_FILE = 'split_dataset/val_ids.txt'
TEST_IDS_FILE = 'split_dataset/test_ids.txt'


def load_split_ids(file_path: str) -> List[str]:
    """Loads image IDs from a text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split ID file not found: {file_path}")
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_annotations(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads the full annotations and maps
    them by image_id for quick lookup."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Annotation file not found: {file_path}")
    with open(file_path, 'rb') as f:
        records = pickle.load(f)

    return {record['image_id']: record for record in records}


class WaterfowlDataset(Dataset):
    def __init__(self, split: str, transforms: T.Compose = None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            Defines which subset of IDs to use.
            transforms (T.Compose, optional): Composed torchvision transforms.
        """
        self.transforms = transforms

        self.all_annotations = load_annotations(INPUT_FILE)

        if split == 'train':
            id_file = TRAIN_IDS_FILE
        elif split == 'val':
            id_file = VAL_IDS_FILE
        elif split == 'test':
            id_file = TEST_IDS_FILE
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        self.image_ids = load_split_ids(id_file)

        self.records = [self.all_annotations[id] for id in self.image_ids]

        print(f"Initialized WaterfowlDataset with {len(self.records)} images for split: {split}.")  # noqa: E501

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        """
        Loads and preprocesses the image and its target annotations using
        the Albumentations pipeline.
        """
        record = self.records[idx]

        img_path = record['file_path']

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise IOError(f"Failed to load image at {img_path}")

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        h, w = img.shape[:2]

        boxes_list = []
        labels_list = []

        if record['has_objects']:
            for ann in record['annotations']:
                bbox = ann['bbox']

                h, w = img.shape[:2]

                x_min = max(0, min(w - 1, bbox[0]))
                y_min = max(0, min(h - 1, bbox[1]))
                x_max = max(0, min(w, bbox[2]))
                y_max = max(0, min(h, bbox[3]))

                if x_max > x_min and y_max > y_min:
                    boxes_list.append([x_min, y_min, x_max, y_max])
                    labels_list.append(ann['category_id'])

        target = {}
        target["image_id"] = torch.tensor([idx])
        target["boxes"] = torch.as_tensor(boxes_list, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels_list, dtype=torch.int64)

        if self.transforms is not None:
            img_tensor, target = self.transforms(img, target)
        else:
            img_tensor = T.functional.to_tensor(img.astype(np.float32))

        return img_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))
