"""
Training loop for GestureCNN.

Loads the gesture dataset, applies augmentation to training data,
trains the CNN with early stopping, and saves the best checkpoint.

Usage:
    python3 classifier/train.py

Output:
    artifacts/models/cnn_best.pt  — best model checkpoint by val loss
    artifacts/models/training_log.json — per-epoch loss and accuracy log
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from classifier.config import (
    HAND_CROP_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    STATIC_GESTURE_CLASSES,
)
from classifier.models.cnn import GestureCNN
from classifier.data.splits import load_splits, get_split_paths
from classifier.data.augmentation import AugmentationPipeline


# Configuration                                                       

LEARNING_RATE = 1e-4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 64
NUM_WORKERS = 0
MANIFEST_PATH = "data/manifest_split.csv"
CHECKPOINT_PATH = "classifier/models/weights/cnn_best.pt"
LOG_PATH = "artifacts/models/training_log.json"


# Dataset                                                             

class GestureTrainDataset(Dataset):
    """
    PyTorch Dataset for gesture images.
    Applies augmentation when augment=True (training only).
    """

    def __init__(
        self,
        samples: list[tuple[str, str]],
        augment: bool = False,
    ):
        self.samples = samples
        self.label_to_idx = {cls: i for i, cls in enumerate(STATIC_GESTURE_CLASSES)}
        self.augment = augment
        self.augmentation = AugmentationPipeline(
            flip_prob=0.5,
            brightness_range=(0.5, 1.5),
            contrast_range=(0.5, 1.5),
            max_rotation_degrees=20.0,
        ) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        label_idx = self.label_to_idx[label]

        # Load and resize
        img = Image.open(img_path).convert("RGB")
        img = img.resize((HAND_CROP_SIZE, HAND_CROP_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # Apply augmentation before normalization on training set only
        if self.augmentation is not None:
            arr = self.augmentation(arr)

        # Normalize
        mean = np.array(NORMALIZATION_MEAN, dtype=np.float32)
        std = np.array(NORMALIZATION_STD, dtype=np.float32)
        arr = (arr - mean) / std

        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor, label_idx


# Training loop                                                       

def train() -> None:
    """Run the full training loop with early stopping."""

    print("=" * 60)
    print("GestureCNN Training")
    print("=" * 60)

    # Load splits
    print("\nLoading dataset splits...")
    splits = load_splits(MANIFEST_PATH)
    train_samples = get_split_paths(splits, "train")
    val_samples = get_split_paths(splits, "val")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")

    # Datasets and loaders
    train_dataset = GestureTrainDataset(train_samples, augment=True)
    val_dataset = GestureTrainDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = GestureCNN(num_classes=len(STATIC_GESTURE_CLASSES)).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Reduce learning rate when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Output dirs
    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Training state
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    log: list[dict] = []

    print(f"\nTraining for up to {MAX_EPOCHS} epochs (patience={EARLY_STOPPING_PATIENCE})...\n")

    for epoch in range(1, MAX_EPOCHS + 1):

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_imgs.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_imgs)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item() * batch_imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_imgs.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Step scheduler based on val loss
        scheduler.step(avg_val_loss)

        # ---- Log ----
        log.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        print(
            f"Epoch {epoch:>3}/{MAX_EPOCHS} | "
            f"Train loss: {avg_train_loss:.4f} acc: {train_acc:.3f} | "
            f"Val loss: {avg_val_loss:.4f} acc: {val_acc:.3f}"
        )

        # ---- Early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ New best checkpoint saved (val loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    # Save training log
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Training log saved to: {LOG_PATH}")


if __name__ == "__main__":
    train()