"""Verify the dataset split and print detailed statistics.

Usage:
    python -m dataset.verify_split

Run this after split_data.py to confirm:
- All samples have a split assignment
- No subject appears in multiple splits (no data leakage)
- Class balance looks reasonable across splits
- All image/video files actually exist on disk
"""

import csv
from collections import defaultdict
from pathlib import Path


def verify_split(manifest_path: Path, project_root: Path) -> bool:
    """Run all verification checks on the split manifest.

    Returns:
        True if all checks pass, False if any issues found.
    """
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found.")
        print("Run these scripts first, in order:")
        print("  1. python -m dataset.download_datasets")
        print("  2. python -m dataset.build_manifest")
        print("  3. python -m dataset.split_data")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("ERROR: Manifest is empty.")
        return False

    all_passed = True

    # Check 1: Every row has a split
    print("CHECK 1: Every sample has a split assignment")
    valid_splits = {"train", "val", "test"}
    missing_split = [r for r in rows if r.get("split") not in valid_splits]
    if missing_split:
        print(f"  FAIL: {len(missing_split)} samples have no valid split")
        all_passed = False
    else:
        print(f"  PASS: All {len(rows)} samples assigned to train/val/test")

    # Check 2: No subject leaks across splits (for Kaggle data)
    print("\nCHECK 2: No subject appears in multiple splits (data leakage check)")

    # Only check leapgestrecog subjects — team videos use random split
    leap_rows = [r for r in rows if r["source"] == "leapgestrecog"]
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for row in leap_rows:
        subject_splits[row["subject"]].add(row["split"])

    leaked = {s: sp for s, sp in subject_splits.items() if len(sp) > 1}
    if leaked:
        print(f"  FAIL: {len(leaked)} subjects appear in multiple splits:")
        for subj, splits in leaked.items():
            print(f"    Subject {subj} is in: {splits}")
        all_passed = False
    elif leap_rows:
        print(f"  PASS: All {len(subject_splits)} LeapGestRecog subjects are in exactly one split")
    else:
        print(f"  SKIP: No LeapGestRecog data to check")

    # Check 3: Class distribution
    print("\nCHECK 3: Class distribution across splits")
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["split"]][row["label"]] += 1

    labels = sorted({row["label"] for row in rows})
    splits = ["train", "val", "test"]

    header = f"  {'Gesture':<18}" + "".join(f"{s:>10}" for s in splits) + f"{'TOTAL':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label in labels:
        row_str = f"  {label:<18}"
        total = 0
        for split in splits:
            count = counts[split][label]
            total += count
            row_str += f"{count:>10}"
        row_str += f"{total:>10}"
        print(row_str)

    # Check for empty cells
    empty_cells = []
    for split in splits:
        for label in labels:
            if counts[split][label] == 0:
                empty_cells.append((split, label))

    if empty_cells:
        print(f"\n  WARNING: {len(empty_cells)} class/split combos have 0 samples:")
        for split, label in empty_cells:
            print(f"    '{label}' has no samples in '{split}'")
        print("  This may be okay for swipe videos (few samples),")
        print("  but every class should ideally be in every split.")
    else:
        print(f"\n  PASS: Every class is represented in every split")

    # Check 4: Files exist on disk
    print(f"\nCHECK 4: Verifying files exist on disk")
    missing_files = []
    for row in rows:
        file_path = project_root / row["filepath"]
        if not file_path.exists():
            missing_files.append(row["filepath"])

    if missing_files:
        print(f"  FAIL: {len(missing_files)} files not found. First 5:")
        for path in missing_files[:5]:
            print(f"    {path}")
        all_passed = False
    else:
        print(f"  PASS: All {len(rows)} files exist")

    # Check 5: Media type summary
    print(f"\nCHECK 5: Media type summary")
    media_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        media_counts[row.get("media_type", "unknown")] += 1
    for media, count in sorted(media_counts.items()):
        print(f"  {media}: {count} samples")

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("ALL CHECKS PASSED — dataset is ready for training!")
    else:
        print("SOME CHECKS FAILED — see details above.")
    print("=" * 50)

    return all_passed


def main() -> None:
    """Entry point for the verification script."""
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "manifest_split.csv"

    print("=" * 60)
    print("Dataset Split Verifier (Module A)")
    print("=" * 60)
    print()

    verify_split(manifest_path, project_root)


if __name__ == "__main__":
    main()
