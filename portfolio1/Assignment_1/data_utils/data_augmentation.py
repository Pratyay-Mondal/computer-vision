import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

TARGET_HEIGHT = 640
TARGET_WIDTH = 640
TARGET_SIZE_TUPLE = (TARGET_HEIGHT, TARGET_WIDTH)

MEAN = 0.4112
STD = 0.1479


class Compose:
    def __init__(self, transforms: A.Compose):
        self.albumentations_pipeline = transforms

    def __call__(self, image: np.ndarray, target: dict):

        bboxes_list = target["boxes"].tolist() if target["boxes"].numel() > 0 else []  # noqa: E501
        labels_list = target["labels"].tolist() if "labels" in target else []
        augmented = self.albumentations_pipeline(
            image=image,
            bboxes=bboxes_list,
            class_labels=labels_list,
        )

        image_tensor = augmented['image']

        new_bboxes_list = augmented['bboxes']
        new_labels_list = augmented['class_labels']

        valid_bboxes = []
        valid_labels = []

        for bbox, label in zip(new_bboxes_list, new_labels_list):
            x_min, y_min, x_max, y_max = bbox

            if x_max > x_min and y_max > y_min:
                valid_bboxes.append(bbox)
                valid_labels.append(label)

        new_bboxes_list = valid_bboxes
        new_labels_list = valid_labels

        if new_bboxes_list:
            new_boxes = torch.tensor(new_bboxes_list, dtype=torch.float32)
            if new_boxes.dim() == 1:
                new_boxes = new_boxes.unsqueeze(0)
        else:
            new_boxes = torch.empty((0, 4), dtype=torch.float32)

        target["boxes"] = new_boxes
        if "labels" in target:
            target["labels"] = torch.tensor(new_labels_list, dtype=target["labels"].dtype)  # noqa: E501

        if "image_id" in target:
            target["image_id"] = augmented['image_id'] if 'image_id' in augmented else target["image_id"]  # noqa: E501

        return image_tensor, target


def get_transform(train: bool) -> Compose:

    if train:
        albumentations_transforms = A.Compose(
            [
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    rotate=(-10, 10),
                    shear={"x": (-5, 5), "y": (-5, 5)},
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=0.7
                ),

                A.RandomResizedCrop(
                    size=TARGET_SIZE_TUPLE,
                    scale=(1.0, 1.0),
                    p=0.5
                ),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),

                A.RandomBrightnessContrast(brightness_limit=0.3,
                                           contrast_limit=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(std_range=(0.012, 0.028),
                             mean_range=(0.0, 0.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),

                A.Normalize(mean=[MEAN], std=[STD], max_pixel_value=255.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc",
                                     label_fields=['class_labels'])
        )
    else:
        albumentations_transforms = A.Compose(
            [
                A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
                A.Normalize(mean=[MEAN], std=[STD], max_pixel_value=255.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc",
                                     label_fields=['class_labels'])
        )
    return Compose(albumentations_transforms)
