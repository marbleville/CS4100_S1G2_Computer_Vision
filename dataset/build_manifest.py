"""Build a manifest CSV cataloging every image/video in the dataset.


This scans two data sources and creates data/manifest.csv:

1. data/raw/leapgestrecog/  — Kaggle static gesture images
   (downloaded by download_datasets.py)
   Only includes gestures listed in gesture_map.STATIC_GESTURES.

2. data/left_swipe/ and data/right_swipe/  — Team-recorded swipe
   videos (recorded by data/video_recorder.py)

The manifest is the single source of truth that all downstream
scripts (splitting, training, evaluation) read from.
"""

import csv
from pathlib import Path

from dataset.gesture_map import STATIC_GESTURES, DYNAMIC_GESTURES


# LeapGestRecog: maps folder names to clean gesture labels
# Only gestures in STATIC_GESTURES are included.
LEAPGESTRECOG_ALL_LABELS = {
    "01_palm": "palm",
    "02_l": "l",
    "03_fist": "fist",
    "04_fist_moved": "fist_moved",
    "05_thumb": "thumb",
    "06_index": "index",
    "07_ok": "ok",
    "08_palm_moved": "palm_moved",
    "09_c": "c",
    "10_down": "down",
}

# Filter to only the gestures we're actually using
LEAPGESTRECOG_LABELS = {
    folder: label
    for folder, label in LEAPGESTRECOG_ALL_LABELS.items()
    if label in STATIC_GESTURES
}

# Team-recorded swipe videos: maps folder names to gesture labels
SWIPE_LABELS = {name: name for name in DYNAMIC_GESTURES}

# File extensions we recognize
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def build_manifest(project_root: Path, output_path: Path) -> int:
    """Scan all data sources and write manifest CSV.

    Args:
        project_root: Root of the project (contains data/).
        output_path: Where to write the manifest CSV.

    Returns:
        Total number of samples cataloged.

    Raises:
        FileNotFoundError: If no data sources are found at all.
    """
    rows: list[dict[str, str]] = []

    # Source 1: LeapGestRecog (Kaggle static images)
    leap_dir = project_root / "data" / "raw" / "leapgestrecog"
    if leap_dir.exists():
        print(f"Scanning LeapGestRecog at {leap_dir}...")
        print(f"  Filtering to active gestures: {sorted(STATIC_GESTURES.keys())}")
        leap_rows = _scan_leapgestrecog(leap_dir, project_root)
        rows.extend(leap_rows)
        print(f"  Found {len(leap_rows)} images")
    else:
        print(f"WARNING: {leap_dir} not found.")
        print("  Run 'python -m dataset.download_datasets' first.")

    # Source 2: Team-recorded swipe videos
    data_dir = project_root / "data"
    for folder_name, label in SWIPE_LABELS.items():
        swipe_dir = data_dir / folder_name
        if swipe_dir.exists():
            print(f"Scanning team swipe videos at {swipe_dir}...")
            swipe_rows = _scan_swipe_videos(swipe_dir, label, project_root)
            rows.extend(swipe_rows)
            print(f"  Found {len(swipe_rows)} videos")

    if not rows:
        raise FileNotFoundError(
            "No data found. Make sure you have either:\n"
            "  - Downloaded LeapGestRecog (python -m dataset.download_datasets)\n"
            "  - Recorded swipe videos (python data/video_recorder.py)"
        )

    # Sort for deterministic ordering
    rows.sort(key=lambda r: r["filepath"])

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "label", "subject", "source", "media_type"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest written to {output_path}")
    print(f"Total samples: {len(rows)}")

    _print_class_summary(rows)

    return len(rows)


def _scan_leapgestrecog(
    leap_dir: Path, project_root: Path
) -> list[dict[str, str]]:
    """Scan the LeapGestRecog folder structure.

    Only includes gesture classes listed in LEAPGESTRECOG_LABELS
    (filtered by STATIC_GESTURES from gesture_map.py).
    Skips unused gestures silently.
    """
    rows = []

    for subject_dir in sorted(leap_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name  # "00", "01", etc.

        for gesture_dir in sorted(subject_dir.iterdir()):
            if not gesture_dir.is_dir():
                continue

            label = LEAPGESTRECOG_LABELS.get(gesture_dir.name)
            if label is None:
                # Either an unknown folder or a gesture we're not using — skip
                continue

            for image_file in sorted(gesture_dir.iterdir()):
                if image_file.suffix.lower() in IMAGE_EXTENSIONS:
                    relative_path = image_file.relative_to(project_root)
                    rows.append({
                        "filepath": str(relative_path),
                        "label": label,
                        "subject": f"leap_{subject_id}",
                        "source": "leapgestrecog",
                        "media_type": "image",
                    })

    return rows


def _scan_swipe_videos(
    swipe_dir: Path, label: str, project_root: Path
) -> list[dict[str, str]]:
    """Scan a team-recorded swipe video folder.

    These videos are recorded by data/video_recorder.py.
    """
    rows = []

    for video_file in sorted(swipe_dir.iterdir()):
        if video_file.suffix.lower() in VIDEO_EXTENSIONS:
            relative_path = video_file.relative_to(project_root)
            rows.append({
                "filepath": str(relative_path),
                "label": label,
                "subject": "team",
                "source": "team_recorded",
                "media_type": "video",
            })

    return rows


def _print_class_summary(rows: list[dict[str, str]]) -> None:
    """Print a summary table of samples per class."""
    label_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1

    print("\nClass Distribution:")
    print("-" * 50)
    print(f"{'Gesture':<18} {'Action':<18} {'Count':>8}")
    print("-" * 50)

    from dataset.gesture_map import ALL_GESTURE_ACTIONS
    for label in sorted(label_counts.keys()):
        action = ALL_GESTURE_ACTIONS.get(label, "?")
        print(f"{label:<18} {action:<18} {label_counts[label]:>8}")
    print("-" * 50)
    print(f"{'TOTAL':<37} {len(rows):>8}")

    print("\nBy Source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} samples")


def main() -> None:
    """Entry point for the manifest builder."""
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "manifest.csv"

    print("=" * 60)
    print("Manifest Builder (Module A)")
    print("=" * 60)
    print()

    build_manifest(project_root, output_path)


if __name__ == "__main__":
    main()
