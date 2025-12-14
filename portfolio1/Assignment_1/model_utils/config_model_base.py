import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


NUM_CLASSES = 1 + 1

CLASS_ID = 1
CLASS_NAME = 'waterfowl'

BACKBONE_NAME = 'resnet50_fpn'
USE_PRETRAINED_BACKBONE = True


def get_model_instance_segmentation(num_classes: int = NUM_CLASSES):
    """
    Loads a pre-trained Faster R-CNN model and modifies the final head
    to fit the number of classes in our dataset (waterfowl + background).

    Args:
        num_classes: The total number of classes (1 object + 1 background).

    Returns:
        A PyTorch Faster R-CNN model.
    """

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT' if USE_PRETRAINED_BACKBONE else None
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.rpn.anchor_generator.sizes = (
        (8, 16, 32, 64, 128),
        (16, 32, 64, 128, 256),
        (32, 64, 128, 256, 512),
        (64, 128, 256, 512, 1024),
        (128, 256, 512, 1024, 2048),
    )

    return model
