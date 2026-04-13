"""
Compute per-channel mean and standard deviation from the training set.

Run this script once after the dataset is downloaded and split to replace
the placeholder normalization constants in classifier/config.py.

Usage:
    python classifier/scripts/compute_normalization.py

Output:
    Prints the computed mean and std values to copy into config.py.
    Also saves results to artifacts/normalization_stats.json.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from classifier.data.splits import load_splits, get_split_paths
from classifier.config import HAND_CROP_SIZE


def compute_normalization_stats(
    manifest_path: str = "data/manifest_split.csv",
    output_path: str = "artifacts/normalization_stats.json",
) -> tuple[list[float], list[float]]:
    """
    Compute per-channel mean and std from the training set only.

    Loads every training image, resizes to HAND_CROP_SIZE, and accumulates
    per-channel statistics using Welford's online algorithm to avoid loading
    the entire dataset into memory at once.

    Args:
        manifest_path: Path to Module A's split manifest CSV.
        output_path:   Where to save the computed stats as JSON.

    Returns:
        (mean, std) as lists of 3 floats, one per RGB channel.
    """
    print("Loading training split from manifest...")
    splits = load_splits(manifest_path)
    train_paths = get_split_paths(splits, "train")

    if not train_paths:
        raise ValueError("No training samples found — check manifest path and label mapping.")

    print(f"Computing stats over {len(train_paths)} training images...")

    # Welford's online algorithm for numerically stable mean and variance
    # Avoids loading all images into memory simultaneously
    count = 0
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)

    for i, (img_path, _) in enumerate(train_paths):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(train_paths)} images...")

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((HAND_CROP_SIZE, HAND_CROP_SIZE), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # shape (H, W, 3)
        except Exception as e:
            print(f"  Warning: skipping {img_path}: {e}")
            continue

        # Per-pixel update — treat each pixel as an independent sample
        pixels = arr.reshape(-1, 3)  # shape (H*W, 3)
        for pixel in pixels:
            count += 1
            delta = pixel - mean
            mean += delta / count
            delta2 = pixel - mean
            M2 += delta * delta2

    if count < 2:
        raise ValueError(f"Not enough valid pixels to compute stats (got {count}).")

    variance = M2 / (count - 1)
    std = np.sqrt(variance)

    mean_list = mean.tolist()
    std_list = std.tolist()

    # Print results for copying into config.py
    print("\n" + "=" * 50)
    print("Normalization stats computed from training set:")
    print(f"  Mean: {[round(v, 4) for v in mean_list]}")
    print(f"  Std:  {[round(v, 4) for v in std_list]}")
    print("=" * 50)
    print("\nCopy these into classifier/config.py:")
    print(f"  NORMALIZATION_MEAN = {[round(v, 4) for v in mean_list]}")
    print(f"  NORMALIZATION_STD  = {[round(v, 4) for v in std_list]}")

    # Save to artifacts/
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": mean_list,
                "std": std_list,
                "num_pixels": count,
                "num_images": len(train_paths),
                "crop_size": HAND_CROP_SIZE,
            },
            f,
            indent=2,
        )
    print(f"\nStats saved to {output_path}")

    return mean_list, std_list


if __name__ == "__main__":
    compute_normalization_stats()