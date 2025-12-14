import os
import pickle
from typing import Any, Dict, List

import cv2
import numpy as np

INPUT_FILE = '../formatted_annotations.pkl'
TRAIN_IDS_DIR = '../split_dataset'
TRAIN_IDS_FILE = os.path.join(TRAIN_IDS_DIR, 'train_ids.txt')


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads the formatted annotation list from the pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_train_ids(file_path: str) -> List[str]:
    """Loads the list of training image IDs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training IDs file not found: {file_path}")
    with open(file_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def create_id_to_record_map(all_records: List[Dict[str, Any]]):
    """Creates a dictionary mapping image_id to the full record."""
    return {record['image_id']: record for record in all_records}


def determine_max_pixel_value(sample_image_path: str) -> float:
    image_np = cv2.imread(sample_image_path, cv2.IMREAD_UNCHANGED)

    if image_np is None:
        raise ValueError(f"Could not load sample image at {sample_image_path} to check bit depth.")  # noqa: E501

    dtype = image_np.dtype

    if dtype == np.uint8:
        max_val = 255.0
        print(f"Sample image dtype is {dtype}. Setting MAX_PIXEL_VALUE = 255.0 (8-bit).")  # noqa: E501
    elif dtype == np.uint16:
        max_val = 65535.0
        print(f"Sample image dtype is {dtype}. Setting MAX_PIXEL_VALUE = 65535.0 (16-bit).")  # noqa: E501
    else:
        raise TypeError(f"Unsupported image dtype: {dtype}. Cannot determine MAX_PIXEL_VALUE.")  # noqa: E501
    print("dtype and max_val:", dtype, max_val)
    return max_val


def calculate_mean_std():
    try:
        all_records = load_data(INPUT_FILE)
        train_ids = load_train_ids(TRAIN_IDS_FILE)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        return

    id_map = create_id_to_record_map(all_records)

    if not train_ids:
        print("Error: Training ID list is empty.")
        return

    sample_id = train_ids[0]
    sample_path = id_map[sample_id]['file_path']

    try:
        MAX_PIXEL_VALUE = determine_max_pixel_value(sample_path)
    except (ValueError, TypeError) as e:
        print(f"FATAL ERROR during bit depth check: {e}")
        return

    sum_pixels = 0.0
    sum_squared_pixels = 0.0
    total_pixels = 0

    print(f"Processing {len(train_ids)} images...")

    for image_id in train_ids:
        record = id_map.get(image_id)
        if record is None:
            continue

        image_path = record['file_path']

        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image_np is None:
            continue

        scaled_image = image_np.astype(np.float32) / MAX_PIXEL_VALUE
        sum_pixels += scaled_image.sum()
        sum_squared_pixels += (scaled_image ** 2).sum()
        total_pixels += scaled_image.size

    if total_pixels == 0:
        print("Error: No pixels processed after attempting to load all images.")  # noqa: E501
        return

    mean = sum_pixels / total_pixels
    mean_of_squares = sum_squared_pixels / total_pixels
    variance = mean_of_squares - (mean ** 2)
    std = np.sqrt(variance)

    print("-" * 50)
    print("Normalization Constants Calculated:")
    final_mean = [round(mean, 4)]
    final_std = [round(std, 4)]

    print(f"    MAX_PIXEL_VALUE Used: {int(MAX_PIXEL_VALUE)}")
    print(f"    Target MEAN: **{final_mean}**")
    print(f"    Target STD:  **{final_std}**")
    print("-" * 50)

    return final_mean, final_std


if __name__ == '__main__':
    calculate_mean_std()
