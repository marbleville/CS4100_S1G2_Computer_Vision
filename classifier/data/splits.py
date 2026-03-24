"""
Split utility for gesture dataset.

Reads Module A's manifest_split.csv and returns train/val/test splits
as structured dicts for use by GestureDataset and the training pipeline.

Does NOT generate splits — Module A owns that responsibility via
dataset/split_data.py. This module is purely a reader and formatter.
"""

import csv
from pathlib import Path

from classifier.config import STATIC_GESTURE_CLASSES

# Maps Module A's LeapGestRecog label names to our canonical gesture labels.
# TODO: CONFIRM these mappings with Module A before training.
# just rename the labels to be the same for consistency
LABEL_MAP: dict[str, str] = {
    "fist":       "fist",
    "fist_moved": "fist",        
    "palm":       "open",        
    "palm_moved": "open",        
    "thumb":      "thumbs_up",   
    "down":       "thumbs_down", 
}


def load_splits(
    manifest_path: str = "data/manifest_split.csv",
) -> dict[str, list[dict]]:
    """
    Load train/val/test splits from Module A's manifest CSV.

    Reads data/manifest_split.csv, filters to static gesture classes only
    (ignoring wave_left and wave_right which belong to Module D), maps
    Module A's label names to our canonical gesture labels, and returns
    a structured dict for use by GestureDataset and the training pipeline.

    Args:
        manifest_path: Path to Module A's split manifest CSV.
                       Default is 'data/manifest_split.csv'.

    Returns:
        dict with keys 'train', 'val', 'test'. Each value is a list of
        {'path': str, 'label': str} dicts with relative file paths.

    Raises:
        FileNotFoundError: If the manifest CSV does not exist.
        ValueError: If the manifest has no rows after filtering to static gestures.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at '{manifest_path}'. "
            "Run 'python -m dataset.build_manifest' and "
            "'python -m dataset.split_data' from Module A first."
        )

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    skipped_labels: set[str] = set()
    skipped_media: int = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip non-image media (videos belong to Module D)
            if row.get("media_type") != "image":
                skipped_media += 1
                continue

            raw_label = row.get("label", "")
            split = row.get("split", "")

            # Map Module A label to our canonical label
            canonical_label = LABEL_MAP.get(raw_label)

            # Skip if label not in our static gesture classes or not mapped
            if canonical_label is None or canonical_label not in STATIC_GESTURE_CLASSES:
                skipped_labels.add(raw_label)
                continue

            # Skip if split column is missing or invalid
            if split not in splits:
                continue

            splits[split].append({
                "path": row["filepath"],
                "label": canonical_label,
            })

    if skipped_labels:
        print(f"Skipped unmapped labels: {sorted(skipped_labels)}")
    if skipped_media > 0:
        print(f"Skipped {skipped_media} non-image samples (videos — handled by Module D)")

    total = sum(len(v) for v in splits.values())
    if total == 0:
        raise ValueError(
            "No static gesture samples found after filtering. "
            "Check that LABEL_MAP matches Module A's label names in the CSV."
        )

    print_split_summary(splits)
    return splits


def get_split_paths(splits: dict, split: str) -> list[tuple[str, str]]:
    """
    Extract (path, label) pairs for a given split name.

    Args:
        splits: Dict returned by load_splits().
        split:  One of 'train', 'val', or 'test'.

    Returns:
        List of (image_path, label_string) tuples.

    Raises:
        ValueError: If split name is not valid.
    """
    valid_splits = {"train", "val", "test"}
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")
    return [(r["path"], r["label"]) for r in splits[split]]


def print_split_summary(splits: dict) -> None:
    """
    Print a human readable summary of split sizes per class.

    Args:
        splits: Dict returned by load_splits().
    """
    print("\n--- Split Summary ---")
    for split_name in ("train", "val", "test"):
        records = splits.get(split_name, [])
        counts: dict[str, int] = {}
        for r in records:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print(f"{split_name:>6}: {len(records)} samples")
        for cls in sorted(counts):
            print(f"         {cls}: {counts[cls]}")
    print("---------------------\n")