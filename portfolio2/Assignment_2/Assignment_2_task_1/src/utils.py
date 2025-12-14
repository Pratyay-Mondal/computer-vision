import os
import gzip
import torch
import json

def get_model_size_mb(model):
    torch.save(model.state_dict(), "temp_model.pth")
    with open("temp_model.pth", "rb") as f_in:
        with gzip.open("temp_model.pth.gz", "wb") as f_out:
            f_out.writelines(f_in)
    compressed_size = os.path.getsize("temp_model.pth.gz") / (1024 * 1024)
    os.remove("temp_model.pth")
    os.remove("temp_model.pth.gz")
    return compressed_size

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def save_text(content, filepath):
    with open(filepath, "w") as f:
        f.write(str(content))
