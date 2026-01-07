"""
CelebA Dataset loader for VAE training.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Union

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class CelebADataset(Dataset):
    """CelebA dataset for VAE training."""
    
    def __init__(
        self,
        root_dir: Path,
        image_size: int = 64,
        limit: Optional[int] = None,
        normalize_to_minus_one: bool = True,
        return_attributes: bool = False,
        attributes_file: Optional[Path] = None
    ):
        """
        Initialize CelebA dataset.
        
        Args:
            root_dir: Path to img_align_celeba directory
            image_size: Size to resize images to
            limit: Maximum number of images to load
            normalize_to_minus_one: If True, normalize to [-1, 1] (for Gaussian),
                                    else normalize to [0, 1] (for Bernoulli)
            return_attributes: If True, return (image, attributes_dict) tuples
            attributes_file: Path to list_attr_celeba.txt (auto-detected if None)
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.normalize_to_minus_one = normalize_to_minus_one
        self.return_attributes = return_attributes
        
        # Build transform pipeline
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        
        if normalize_to_minus_one:
            transform_list.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
        
        self.transform = transforms.Compose(transform_list)
        
        # Load file names
        self.file_names = sorted([
            f for f in os.listdir(root_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if limit is not None:
            self.file_names = self.file_names[:limit]
        
        # Load attributes if requested
        self.attributes: Optional[Dict[str, Dict[str, int]]] = None
        self.attribute_names: List[str] = []
        
        if return_attributes:
            self._load_attributes(attributes_file)
        
        print(f"[DATASET] Loaded {len(self.file_names)} images from {root_dir}")
        if self.attributes:
            print(f"[DATASET] Loaded {len(self.attribute_names)} attributes")
    
    def _load_attributes(self, attributes_file: Optional[Path] = None):
        """Load CelebA attribute annotations."""
        # Try to find the attributes file
        if attributes_file is None:
            # Check common locations
            possible_paths = [
                self.root_dir.parent / "list_attr_celeba.txt",
                self.root_dir.parent / "Anno" / "list_attr_celeba.txt",
                self.root_dir / ".." / "list_attr_celeba.txt",
                self.root_dir / ".." / "Anno" / "list_attr_celeba.txt",
            ]
            
            for path in possible_paths:
                if path.exists():
                    attributes_file = path
                    break
        
        if attributes_file is None or not Path(attributes_file).exists():
            print("[WARNING] Could not find list_attr_celeba.txt, attributes disabled")
            self.return_attributes = False
            return
        
        print(f"[DATASET] Loading attributes from {attributes_file}")
        
        self.attributes = {}
        
        with open(attributes_file, 'r') as f:
            # First line: number of images
            num_images = int(f.readline().strip())
            
            # Second line: attribute names
            self.attribute_names = f.readline().strip().split()
            
            # Remaining lines: image_name attr1 attr2 ...
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                img_name = parts[0]
                # Attributes are -1 or 1, convert to 0 or 1
                attrs = {
                    name: (1 if int(val) == 1 else 0)
                    for name, val in zip(self.attribute_names, parts[1:])
                }
                self.attributes[img_name] = attrs
    
    def get_attribute_names(self) -> List[str]:
        """Return list of available attribute names."""
        return self.attribute_names.copy()
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx) -> Union[torch.Tensor, tuple]:
        file_name = self.file_names[idx]
        path = self.root_dir / file_name
        img = Image.open(path).convert('RGB')
        img_tensor = self.transform(img)
        
        if self.return_attributes and self.attributes:
            attrs = self.attributes.get(file_name, {})
            return img_tensor, attrs
        
        return img_tensor


def create_dataloaders(
    dataset_dir: Path,
    image_size: int = 64,
    batch_size: int = 128,
    limit: Optional[int] = None,
    train_split: float = 0.9,
    num_workers: int = 2,
    pin_memory: bool = True,
    normalize_to_minus_one: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        dataset_dir: Path to img_align_celeba directory
        image_size: Size to resize images to
        batch_size: Batch size for dataloaders
        limit: Maximum number of images to load
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        normalize_to_minus_one: If True, normalize to [-1, 1], else [0, 1]
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset = CelebADataset(
        root_dir=dataset_dir,
        image_size=image_size,
        limit=limit,
        normalize_to_minus_one=normalize_to_minus_one
    )
    
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"[DATASET] Train samples: {train_size}")
    print(f"[DATASET] Test samples: {test_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"[DATASET] Train batches: {len(train_loader)}")
    print(f"[DATASET] Test batches: {len(test_loader)}")
    
    return train_loader, test_loader
