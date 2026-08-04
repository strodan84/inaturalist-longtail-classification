from typing import Tuple, List
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_dataloaders(
    data_dir: str = "./data", 
    batch_size: int = 128, 
    img_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Generates a fast synthetic dataset in RAM for pipeline debugging.
    Consumes 0 MB disk and makes 0 web requests.
    """
    print("--> Generating Fast Synthetic Dataset (0 MB disk, Instant Load)...")
    
    num_train_samples = 2500
    num_val_samples = 500
    num_classes = 10000

    # Synthetic RGB images matching ResNet input dimensions (N, C, H, W)
    x_train = torch.randn(num_train_samples, 3, img_size, img_size)
    y_train = torch.randint(0, num_classes, (num_train_samples,))
    
    x_val = torch.randn(num_val_samples, 3, img_size, img_size)
    y_val = torch.randint(0, num_classes, (num_val_samples,))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    # Class frequency distribution vector for SeesawLoss / Long-Tail Loss setup
    cls_num_list = [10] * num_classes

    print(f"✅ Fast Synthetic DataLoaders Ready! Training on {num_train_samples} samples across {num_classes} classes.")
    return train_loader, val_loader, cls_num_list