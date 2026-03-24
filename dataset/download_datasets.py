"""Download gesture recognition datasets from Kaggle.

Usage:
    python -m dataset.download_datasets

This script downloads the LeapGestRecog dataset and copies it into
data/raw/leapgestrecog/. only need to run this once

Prerequisites:
    - pip install kagglehub
    - Kaggle API token at ~/.kaggle/kaggle.json
      (See dataset/README.md for setup instructions)
"""

import shutil
from pathlib import Path


def download_leapgestrecog(destination: Path) -> None:
    """Download LeapGestRecog dataset from Kaggle.

    The dataset contains 20,000 infrared hand gesture images:
    - 10 gesture classes
    - 10 subjects (5 male, 5 female)
    - 200 images per gesture per subject
    """
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed.")
        print("Run:  pip install kagglehub")
        print("  or: uv pip install kagglehub")
        return

    print("Downloading LeapGestRecog from Kaggle...")
    print("(This may take a while)\n")

    # kagglehub downloads and caches the dataset, returns the local path
    cached_path = kagglehub.dataset_download("gti-upm/leapgestrecog")
    cached_path = Path(cached_path)

    print(f"Downloaded to cache: {cached_path}")

    # The cached download may have a nested folder structure.
    # We need to find where the subject folders (00, 01, ...) actually are.
    source_dir = _find_subject_folders(cached_path)

    if source_dir is None:
        print(f"ERROR: Could not find subject folders (00, 01, ...) in {cached_path}")
        print("Please check the download and manually copy the data.")
        return

    # Copy to our project's data/raw/ folder
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Copying dataset to {destination}...")
    for item in sorted(source_dir.iterdir()):
        if item.is_dir():
            dest_item = destination / item.name
            if dest_item.exists():
                print(f"  Skipping {item.name}/ (already exists)")
            else:
                shutil.copytree(item, dest_item)
                print(f"  Copied {item.name}/")

    print("\nDone! LeapGestRecog is ready at", destination)


def _find_subject_folders(root: Path) -> Path | None:
    """Find the directory containing subject folders (00, 01, ..., 09).

    Kaggle downloads sometimes have extra nesting, so we search
    for the directory that contains folders named '00' through '09'.
    """
    # Check if root itself contains subject folders
    if _has_subject_folders(root):
        return root

    # Search one or two levels deep
    for child in root.iterdir():
        if child.is_dir():
            if _has_subject_folders(child):
                return child
            for grandchild in child.iterdir():
                if grandchild.is_dir() and _has_subject_folders(grandchild):
                    return grandchild

    return None


def _has_subject_folders(directory: Path) -> bool:
    """Check if a directory contains numbered subject folders."""
    children = {item.name for item in directory.iterdir() if item.is_dir()}
    # LeapGestRecog has folders named "00" through "09"
    return "00" in children and "01" in children


def main() -> None:
    """Entry point for the download script."""
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw" / "leapgestrecog"

    print("=" * 60)
    print("Dataset Downloader (Module A)")
    print("=" * 60)
    print()

    if raw_data_dir.exists() and any(raw_data_dir.iterdir()):
        print(f"LeapGestRecog already exists at {raw_data_dir}")
        print("To re-download, delete that folder and run again.")
    else:
        download_leapgestrecog(raw_data_dir)

    print()
    print("NOTE: Team-recorded swipe videos (data/left_swipe/,")
    print("data/right_swipe/) are managed separately via")
    print("data/video_recorder.py — no download needed for those.")


if __name__ == "__main__":
    main()
