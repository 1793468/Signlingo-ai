"""
Training/evaluation loop for the BiLSTM + Attention sign classifier.

Usage:
    python -m src.training.train --config configs/config.yaml
"""
import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.datasets.sign_dataset import SignDataset
from src.models.lstm_attention import SignLSTMAttention


def build_model(cfg, num_classes, device):
    return SignLSTMAttention(
        num_classes=num_classes,
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad()

            logits, _ = model(X)

            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return total_loss, correct / total


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"

    train_ds = SignDataset(f"{cfg['data']['processed_dir']}/train", augment=True)
    val_ds = SignDataset(f"{cfg['data']['processed_dir']}/val")
    test_ds = SignDataset(f"{cfg['data']['processed_dir']}/test")

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)

    num_classes = cfg["data"]["max_signs"]
    model = build_model(cfg, num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    for epoch in range(cfg["training"]["epochs"]):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        _, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

    _, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"Test accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), f"models/{cfg['model']['name']}_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
