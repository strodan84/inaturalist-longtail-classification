import json
import urllib.request
from io import BytesIO
from typing import Tuple, List

import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image


class WebStreamINatDataset(IterableDataset):
    """
    Downloads image URLs directly from iNaturalist JSON annotations
    and fetches images dynamically into RAM during training.
    """
    def __init__(self, annotation_url: str, transform=None, max_samples: int = 5000):
        super().__init__()
        self.transform = transform
        self.max_samples = max_samples

        print("--> Fetching iNaturalist JSON metadata...")
        req = urllib.request.urlopen(annotation_url)
        data = json.loads(req.read().decode('utf-8'))

        # Map image ID to category ID
        self.samples = []
        for ann in data['annotations'][:max_samples]:
            img_info = next((img for img in data['images'] if img['id'] == ann['image_id']), None)
            if img_info:
                self.samples.append((img_info['coco_url'], ann['category_id']))

    def __iter__(self):
        for url, category_id in self.samples:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    img_bytes = response.read()
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                yield image, category_id
            except Exception:
                continue


def get_dataloaders(
    data_dir: str = "./data", 
    batch_size: int = 128, 
    img_size: int = 224,
    num_workers: int = 0
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

    train_json_url = "https://inaturalist-open-data.s3.amazonaws.com/metadata/2021_train_mini.json"
    val_json_url = "https://inaturalist-open-data.s3.amazonaws.com/metadata/2021_val.json"

    train_dataset = WebStreamINatDataset(train_json_url, transform=train_transform, max_samples=25000)
    val_dataset = WebStreamINatDataset(val_json_url, transform=val_transform, max_samples=2500)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    num_classes = 10000
    cls_num_list = [50] * num_classes

    print("✅ Web-Streaming DataLoaders Initialized! Disk space used: ~0 MB.")
    return train_loader, val_loader, cls_num_list