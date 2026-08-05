import os
import argparse
import torch
import torch.nn as nn
import timm
from typing import Tuple

# Import dataloader generator (using dataset_test for fast verification)
from src.dataset_test import get_dataloaders


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> list[float]:
    """
    Computes top-k accuracy for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions along class dimension
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.item() / batch_size) * 100.0)
        return res


def evaluate(
    model_name: str,
    checkpoint_path: str,
    num_classes: int = 10000,
    batch_size: int = 128,
    device: str = "cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    # 1. Load DataLoaders
    _, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # 2. Build Model Backbone
    print(f"--> Loading backbone architecture: {model_name}")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # 3. Load Saved Weights from Google Drive Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at path: {checkpoint_path}")

    print(f"--> Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle state_dict key matching (whether saved full checkpoint or state_dict directly)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 4. Evaluation Loop
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    print("--> Running validation loop...")
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            batch_top1, batch_top5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size_actual = targets.size(0)

            total_top1 += batch_top1 * batch_size_actual
            total_top5 += batch_top5 * batch_size_actual
            total_samples += batch_size_actual

    top1_acc = total_top1 / total_samples
    top5_acc = total_top5 / total_samples

    print("\n" + "=" * 45)
    print("           EVALUATION RESULTS              ")
    print("=" * 45)
    print(f" Total Validation Samples : {total_samples}")
    print(f" Top-1 Accuracy           : {top1_acc:.2f}%")
    print(f" Top-5 Accuracy           : {top5_acc:.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate iNaturalist Model Checkpoint")
    parser.add_argument("--model-name", type=str, default="resnet18", help="timm model architecture")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/content/drive/MyDrive/inat_checkpoints/resnet18_best.pt",
        help="Path to saved model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10000)

    args = parser.parse_args()
    evaluate(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
    )