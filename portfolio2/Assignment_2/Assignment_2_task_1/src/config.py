experiments_config = [
    {"name": "OneShot_L1_50", "method": "unstructured_l1", "target_sparsity": 0.5, "steps": 1},
    {"name": "OneShot_L1_90", "method": "unstructured_l1", "target_sparsity": 0.9, "steps": 1},
    {"name": "OneShot_Random_90", "method": "unstructured_random", "target_sparsity": 0.9, "steps": 1},
    {"name": "Iterative_L1_90", "method": "unstructured_l1", "target_sparsity": 0.9, "steps": 9},
    {"name": "Iterative_Structured_90", "method": "structured_l2", "target_sparsity": 0.9, "steps": 5},
]

MODEL_PATH = './mobilenet_v2_cifar10.pth'
OUTPUT_DIR = './outputs'
DATA_DIR = './data'
