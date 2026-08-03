from typing import Tuple, List
import platform
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import INaturalist


def get_dataloaders(
    data_dir: str = "./data", 
    batch_size: int = 32, 
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Downloads and configures iNaturalist 2021 DataLoaders.
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

    train_dataset = INaturalist(
        root=data_dir, 
        version="2017",  # Switched from 2021_train_mini to 2017
        transform=train_transform, 
        download=True
    )
        
    val_dataset = INaturalist(
        root=data_dir, 
        version="2017", 
        transform=val_transform, 
        download=True
    )

    # Automatically set zero worker processes on Windows to prevent CPU freeze
    num_workers = 0 if platform.system() == "Windows" else 2
    is_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=is_cuda
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=is_cuda
    )

    # Compute per-class sample frequencies
    cls_num_list = [0] * len(train_dataset.all_categories)
    for target_id, _ in train_dataset.index:
        cls_num_list[target_id] += 1

    return train_loader, val_loader, cls_num_list