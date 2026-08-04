from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import INaturalist


def get_dataloaders(
    data_dir: str = "./data", 
    batch_size: int = 128, 
    img_size: int = 224,
    num_workers: int = 2,
    subset_ratio: float = 0.05  # Use 5% of dataset (~25k samples) for fast execution
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Downloads iNaturalist 2021 and uses a subset to prevent high GPU/CPU compute times.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("--> Downloading/Loading iNaturalist 2021 (Mini)...")
    full_train = INaturalist(root=data_dir, version="2021_train_mini", transform=train_transform, download=True)
    full_val = INaturalist(root=data_dir, version="2021_valid", transform=val_transform, download=True)

    # Subsample indices for fast debugging
    train_size = int(len(full_train) * subset_ratio)
    val_size = int(len(full_val) * subset_ratio)

    train_indices = list(range(train_size))
    val_indices = list(range(val_size))

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )

    # Compute class distribution from full dataset (10,000 classes)
    num_classes = len(full_train.all_categories) if hasattr(full_train, 'all_categories') else 10000
    cls_num_list = [0] * num_classes

    # Count occurrences in the selected subset
    for i in train_indices:
        target_id = full_train.index[i][0]
        cls_num_list[target_id] += 1

    print(f"✅ DataLoaders Ready! Training on {train_size} images across {num_classes} categories.")
    return train_loader, val_loader, cls_num_list