from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import INaturalist


def get_dataloaders(
    data_dir: str = "./data", 
    batch_size: int = 32, 
    img_size: int = 224,
    num_workers: int = 4
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
        version="2021_train_mini", 
        transform=train_transform, 
        download=True
    )
    val_dataset = INaturalist(
        root=data_dir, 
        version="2021_valid", 
        transform=val_transform, 
        download=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # Compute per-class instance counts for Seesaw Loss
    cls_num_list = [0] * len(train_dataset.categories)
    for _, target in train_dataset.index:
        cls_num_list[target] += 1

    return train_loader, val_loader, cls_num_list