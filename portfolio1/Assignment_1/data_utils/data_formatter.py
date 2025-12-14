import glob
import os
import pickle

import pandas as pd

ROOT_DIR = 'dataset'

POS_IMG_DIR = os.path.join(ROOT_DIR, 'data', 'imagery_dataset',
                           'images_ground_truth', 'positive_images')

NEG_IMG_DIR = os.path.join(ROOT_DIR, 'data', 'imagery_dataset',
                           'images_ground_truth', 'negative_images')

LABEL_FOLDER = os.path.join(ROOT_DIR, 'data', 'imagery_dataset',
                            'images_ground_truth', 'ground_truth_labels')

OUTPUT_FILE = 'formatted_annotations.pkl'


def find_label_csv(label_folder: str) -> str:
    """Finds the first CSV file in the label folder."""
    csv_files = glob.glob(os.path.join(label_folder, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in the label folder: {label_folder}")  # noqa: E501
    print(f"Found label CSV: {os.path.basename(csv_files[0])}")
    return csv_files[0]


def parse_thermal_annotations(df: pd.DataFrame):

    df = df.rename(columns={
        'imageFilename': 'filename',
        'x(column)': 'xmin',
        'y(row)': 'ymin'
    })

    df['xmax'] = df['xmin'] + df['width']
    df['ymax'] = df['ymin'] + df['height']

    df['category_id'] = 1

    formatted_data = []

    for filename, group in df.groupby('filename'):
        full_path = os.path.join(POS_IMG_DIR, filename)

        annotations = []
        for index, row in group.iterrows():
            annotations.append({
                'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                'category_id': row['category_id'],
                'category_name': 'waterfowl'
            })

        formatted_data.append({
            'image_id': filename,
            'file_path': full_path,
            'annotations': annotations,
            'has_objects': True
        })

    return formatted_data


def process_negative_images():
    """
    Creates records for negative images.
    """
    print(f"Processing negative images from: {NEG_IMG_DIR}")

    negative_records = []
    for filename in os.listdir(NEG_IMG_DIR):
        if filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg')):
            full_path = os.path.join(NEG_IMG_DIR, filename)
            negative_records.append({
                'image_id': filename,
                'file_path': full_path,
                'annotations': [],
                'has_objects': False
            })

    return negative_records


def run_data_parser():
    try:
        label_csv_path = find_label_csv(LABEL_FOLDER)
        raw_df = pd.read_csv(label_csv_path)

        print(f"Found {len(raw_df)} total annotations. Processing positive images...")  # noqa: E501
        positive_records = parse_thermal_annotations(raw_df)
        print(f"Formatted {len(positive_records)} positive images.")

        negative_records = process_negative_images()
        print(f"Formatted {len(negative_records)} negative images.")

        all_records = positive_records + negative_records

        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(all_records, f)

        print("-" * 50)
        print(f"Total images parsed and formatted: {len(all_records)}")
        print(f"Formatted data saved to: {OUTPUT_FILE}")

    except FileNotFoundError as e:
        print("ERROR: A required file or directory was not found.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
