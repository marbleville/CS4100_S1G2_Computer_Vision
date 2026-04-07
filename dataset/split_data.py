"""Split the dataset manifest into train/val/test sets.

Usage:
    python -m dataset.split_data

This reads data/manifest.csv and creates data/manifest_split.csv
with an added 'split' column (train, val, or test).

Key design decisions:

    1. STRATIFIED by label — each gesture class has roughly the same
       proportion in every split.

    2. GROUPED by subject — all images from a given subject land in
       the SAME split.

    3. REPRODUCIBLE — uses a fixed random seed (42) so every
       teammate gets the exact same split when they run this script.

    4. MIXED MEDIA — handles both Kaggle images (many subjects) and
       team-recorded swipe videos (few samples). The swipe videos
       have too few samples for subject-based splitting, so they
       use a simple random split instead.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path


# CONFIGURATION — change these values

TRAIN_RATIO = 0.70   # 70% for training
VAL_RATIO = 0.15     # 15% for validation
TEST_RATIO = 0.15    # 15% for testing
RANDOM_SEED = 42     # Fixed seed for reproducibility


def split_manifest(manifest_path: Path, output_path: Path) -> None:
    """Read manifest CSV, assign splits, write output CSV.

    Splitting strategy depends on how much data we have per source:

    - LeapGestRecog (10 subjects, 20k images): We split at the SUBJECT
      level. Subjects are shuffled and assigned to train/val/test.
      All images from a subject stay together.

    - Team swipe videos (small count): We do a simple random split
      at the individual sample level, since we don't have enough
      samples or subjects for grouped splitting.

    Raises:
        FileNotFoundError: If the manifest file doesn't exist.
        ValueError: If the manifest is empty.
    """

    # Step 1: Read the manifest
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run build_manifest.py first."
        )

    rows = _read_manifest(manifest_path)
    if not rows:
        raise ValueError(
            "Manifest is empty. Run build_manifest.py first."
        )

    print(f"Read {len(rows)} samples from {manifest_path}")

    # Step 2: Separate by source so we can split appropriately
    leapgest_rows = [r for r in rows if r["source"] == "leapgestrecog"]
    team_rows = [r for r in rows if r["source"] == "team_recorded"]
    other_rows = [r for r in rows if r["source"] not in ("leapgestrecog", "team_recorded")]

    rng = random.Random(RANDOM_SEED)

    # Step 3a: Split LeapGestRecog by SUBJECT (grouped)
    if leapgest_rows:
        print(f"\nSplitting LeapGestRecog ({len(leapgest_rows)} images) by subject...")
        _split_by_subject(leapgest_rows, rng)

    # Step 3b: Split team swipe videos randomly
    if team_rows:
        print(f"\nSplitting team-recorded videos ({len(team_rows)} videos) randomly...")
        _split_randomly(team_rows, rng)

    # Step 3c: Any future sources — split randomly
    if other_rows:
        print(f"\nSplitting other data ({len(other_rows)} samples) randomly...")
        _split_randomly(other_rows, rng)

    # Step 4: Combine and write output
    all_rows = leapgest_rows + team_rows + other_rows
    all_rows.sort(key=lambda r: r["filepath"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "label", "subject", "source", "media_type", "split"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSplit manifest written to {output_path}")

    _print_split_summary(all_rows)


def _split_by_subject(rows: list[dict[str, str]], rng: random.Random) -> None:
    """Assign splits based on subject grouping.

    How this works:
    1. Get list of all unique subjects
    2. Shuffle them (with fixed seed for reproducibility)
    3. First 70% of subjects → train
    4. Next 15% → val
    5. Last 15% → test
    6. Every image from a subject goes to that subject's split

    Without this, the model could see the same person's hand in
    training AND in test, which makes test accuracy misleadingly high.
    """
    # Find all subjects
    subjects = sorted({r["subject"] for r in rows})
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val = max(1, round(n * VAL_RATIO))

    # Make sure test gets at least 1 subject if possible
    if n >= 3:
        n_test = n - n_train - n_val
        if n_test < 1:
            n_train = n - 2
            n_val = 1
    else:
        # Very few subjects — just put them all in train
        n_train = n
        n_val = 0

    train_subjects = set(shuffled[:n_train])
    val_subjects = set(shuffled[n_train:n_train + n_val])
    test_subjects = set(shuffled[n_train + n_val:])

    print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"  Val subjects   ({len(val_subjects)}):   {sorted(val_subjects)}")
    print(f"  Test subjects  ({len(test_subjects)}):  {sorted(test_subjects)}")

    # Assign split to each row
    for row in rows:
        subj = row["subject"]
        if subj in train_subjects:
            row["split"] = "train"
        elif subj in val_subjects:
            row["split"] = "val"
        else:
            row["split"] = "test"


def _split_randomly(rows: list[dict[str, str]], rng: random.Random) -> None:
    """Assign splits randomly for small datasets without subject structure.

    Used for team-recorded swipe videos where we have too few samples
    for subject-based grouping.
    """
    shuffled_indices = list(range(len(rows)))
    rng.shuffle(shuffled_indices)

    n = len(shuffled_indices)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val = max(1, round(n * VAL_RATIO))

    train_indices = set(shuffled_indices[:n_train])
    val_indices = set(shuffled_indices[n_train:n_train + n_val])

    train_count = 0
    val_count = 0
    test_count = 0

    for i, row in enumerate(rows):
        if i in train_indices:
            row["split"] = "train"
            train_count += 1
        elif i in val_indices:
            row["split"] = "val"
            val_count += 1
        else:
            row["split"] = "test"
            test_count += 1

    print(f"  Train: {train_count}, Val: {val_count}, Test: {test_count}")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Read manifest CSV into a list of dicts.

    Raises:
        FileNotFoundError: If the manifest file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_manifest.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _print_split_summary(rows: list[dict[str, str]]) -> None:
    """Print a detailed summary table of the split distribution."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["split"]][row["label"]] += 1

    labels = sorted({row["label"] for row in rows})
    splits = ["train", "val", "test"]

    print("\nSplit Distribution:")
    header = f"{'Gesture':<18}" + "".join(f"{s:>10}" for s in splits) + f"{'TOTAL':>10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for label in labels:
        row_str = f"{label:<18}"
        total = 0
        for split in splits:
            count = counts[split][label]
            total += count
            row_str += f"{count:>10}"
        row_str += f"{total:>10}"
        print(row_str)

    print("-" * len(header))
    totals_str = f"{'TOTAL':<18}"
    grand_total = 0
    for split in splits:
        split_total = sum(counts[split].values())
        grand_total += split_total
        totals_str += f"{split_total:>10}"
    totals_str += f"{grand_total:>10}"
    print(totals_str)

    pct_str = f"{'PERCENT':<18}"
    for split in splits:
        split_total = sum(counts[split].values())
        pct = (split_total / grand_total * 100) if grand_total > 0 else 0
        pct_str += f"{pct:>9.1f}%"
    pct_str += f"{'100.0%':>10}"
    print(pct_str)


def main() -> None:
    """Entry point for the split script."""
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "manifest.csv"
    output_path = project_root / "data" / "manifest_split.csv"

    print("=" * 60)
    print("Dataset Splitter (Module A)")
    print("=" * 60)
    print()

    split_manifest(manifest_path, output_path)


if __name__ == "__main__":
    main()
