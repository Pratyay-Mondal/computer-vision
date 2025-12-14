import os
import pickle
import random
from typing import Any, Dict, List

INPUT_FILE = 'formatted_annotations.pkl'

TRAIN_IDS_DIR = 'split_dataset'
TRAIN_IDS_FILE = os.path.join(TRAIN_IDS_DIR, 'train_ids.txt')
VAL_IDS_FILE = os.path.join(TRAIN_IDS_DIR, 'val_ids.txt')
TEST_IDS_FILE = os.path.join(TRAIN_IDS_DIR, 'test_ids.txt')

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RANDOM_SEED = 42


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads the formatted annotation list from the pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}. Please run data_parser_formatter.py first.")  # noqa: E501

    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_ids(ids: List[str], file_path: str):
    """Saves a list of image IDs to a text file, one ID per line."""
    with open(file_path, 'w') as f:
        f.write('\n'.join(ids))
    print(f"Saved {len(ids)} IDs to {file_path}")


def run_data_splitter():
    """
    Loads data, shuffles the IDs, splits them into train/val/test sets,
    ensures the output directory exists, and saves the IDs to separate files.
    """
    try:
        if not os.path.exists(TRAIN_IDS_DIR):
            os.makedirs(TRAIN_IDS_DIR, exist_ok=True)
            print(f"Created directory: {TRAIN_IDS_DIR}")

        random.seed(RANDOM_SEED)

        all_records = load_data(INPUT_FILE)

        print(f"Loaded {len(all_records)} total thermal images (positive + negative).")  # noqa: E501

        all_ids = [record['image_id'] for record in all_records]

        random.shuffle(all_ids)

        total_size = len(all_ids)

        train_end_idx = int(total_size * TRAIN_RATIO)
        val_end_idx = train_end_idx + int(total_size * VAL_RATIO)

        train_ids = all_ids[:train_end_idx]
        val_ids = all_ids[train_end_idx:val_end_idx]
        test_ids = all_ids[val_end_idx:]

        if len(train_ids) + len(val_ids) + len(test_ids) != total_size:
            print("Error: Split sizes do not match total size.")
            return

        print("-" * 50)
        print(f"Split Ratios: Train ({TRAIN_RATIO*100:.0f}%), Val ({VAL_RATIO*100:.0f}%), Test ({100 - TRAIN_RATIO*100 - VAL_RATIO*100:.0f}%)")  # noqa: E501
        print(f"Total Records: {total_size}")
        print(f"Train Set Size: {len(train_ids)}")
        print(f"Validation Set Size: {len(val_ids)}")
        print(f"Test Set Size: {len(test_ids)}")

        save_ids(train_ids, TRAIN_IDS_FILE)
        save_ids(val_ids, VAL_IDS_FILE)
        save_ids(test_ids, TEST_IDS_FILE)

        print("-" * 50)
        print("Data splitting complete. IDs saved.")

    except Exception as e:
        print(f"An error occurred during splitting: {e}")
