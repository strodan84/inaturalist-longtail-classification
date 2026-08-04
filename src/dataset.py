from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


class HFToPyTorchDataset(Dataset):
    """
    Adapts Hugging Face iNaturalist dataset to PyTorch format with transforms.
    """
    def __init__(self, hf_ds, transform=None):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        target = item["label"]  # Integer class index (0 - 9999)

        if self.transform:
            image = self.transform(image)

        return image, target


def get_dataloaders(
    data_dir: str = "./data", 
    batch_size: int = 128, 
    img_size: int = 224,
    num_workers: int = 2,
    train_pct: int = 5  # Percentage of train dataset to download (e.g., 5% = ~25k images)
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Streams a fast, lightweight subset of iNaturalist 2021 via Hugging Face.
    Prevents large 44GB+ tarball downloads and disk crashes.
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

    print(f"--> Loading {train_pct}% slice of iNaturalist 2021_mini from Hugging Face...")
    
    # Load lightweight subsets
    raw_train = load_dataset("inaturalist/inat2021", split=f"train[:{train_pct}%]")
    raw_val = load_dataset("inaturalist/inat2021", split="validation[:5%]")

    # Wrap in PyTorch Datasets
    train_dataset = HFToPyTorchDataset(raw_train, transform=train_transform)
    val_dataset = HFToPyTorchDataset(raw_val, transform=val_transform)

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

    # Compute class frequencies for Seesaw Loss (10,000 total classes in 2021)
    num_classes = 10000
    cls_num_list = [0] * num_classes

    # Count occurrences in the sampled subset
    labels = raw_train["label"]
    for lbl in labels:
        cls_num_list[lbl] += 1

    print(f"✅ DataLoaders Ready! Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader, cls_num_list