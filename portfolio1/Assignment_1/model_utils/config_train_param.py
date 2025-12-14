import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = 'checkpoints/'
MODEL_NAME_PREFIX = 'faster_rcnn_thermal'

BATCH_SIZE = 4
NUM_WORKERS = 4

NUM_EPOCHS = 1
BASE_LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LOG_FREQ = 25

OPTIMIZER_CHOICE = 'SGD'
LR_SCHEDULER_CHOICE = 'StepLR'

LR_STEP_SIZE = 5
LR_GAMMA = 0.1
