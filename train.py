import argparse
import os
import torch
import timm
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.losses import SeesawLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch models on iNaturalist with Seesaw Loss")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--model-name", type=str, default="convnext_small", help="timm model backbone architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--output-dir", type=str, default="./models", help="Directory to save checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    # 1. Load Data
    print("--> Loading iNaturalist DataLoaders...")
    train_loader, val_loader, cls_num_list = get_dataloaders(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        img_size=args.img_size
    )
    num_classes = len(cls_num_list)

    # 2. Build Model
    print(f"--> Creating backbone: {args.model_name}")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # 3. Loss & Optimizer
    criterion = SeesawLoss(cls_num_list=cls_num_list).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(args.output_dir, f"{args.model_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    print("--> Training Complete!")


if __name__ == "__main__":
    main()