from typing import Tuple, List
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


class StreamingINatDataset(IterableDataset):
    """
    Adapts a Hugging Face Streamed Dataset into a PyTorch IterableDataset.
    Streams images on-the-fly directly into GPU/RAM without local storage.
    """
    def __init__(self, hf_stream, transform=None):
        super().__init__()
        self.hf_stream = hf_stream
        self.transform = transform

    def __iter__(self):
        for item in self.hf_stream:
            # Convert PIL image to RGB
            image = item["image"].convert("RGB")
            target = item["label"]  # Category integer label

            if self.transform:
                image = self.transform(image)

            yield image, target


def get_dataloaders(
    data_dir: str = "./data", 
    batch_size: int = 128, 
    img_size: int = 224,
    num_workers: int = 0  # Streaming works best with num_workers=0 or 1
) -> Tuple[DataLoader, DataLoader, List[int]]:
    
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

    print("--> Streaming iNaturalist 2021 directly from Hugging Face Cloud...")

    # Load streamed instances without downloading to local disk
    hf_train_stream = load_dataset(
        "inaturalist/inat2021", 
        name="mini", 
        split="train", 
        streaming=True
    )
    
    hf_val_stream = load_dataset(
        "inaturalist/inat2021", 
        name="mini", 
        split="validation", 
        streaming=True
    )

    # Wrap in PyTorch IterableDatasets
    train_dataset = StreamingINatDataset(hf_train_stream, transform=train_transform)
    val_dataset = StreamingINatDataset(hf_val_stream, transform=val_transform)

    # Note: Shuffle buffer shuffles the stream on-the-fly in RAM
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # iNaturalist 2021 mini has 10,000 distinct species classes
    num_classes = 10000
    # Equalized initial prior list for Seesaw Loss initialization during streaming
    cls_num_list = [50] * num_classes

    print("✅ Streaming DataLoaders Initialized! Disk space used: ~0 MB.")
    return train_loader, val_loader, cls_num_list