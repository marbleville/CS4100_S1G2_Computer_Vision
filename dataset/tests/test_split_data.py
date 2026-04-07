"""Tests for dataset splitting logic."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from dataset.split_data import split_manifest, _split_by_subject, _split_randomly


def _make_leapgest_rows(
    subjects: list[str], labels: list[str], per_combo: int = 5
) -> list[dict[str, str]]:
    """Generate fake manifest rows for LeapGestRecog data."""
    rows = []
    for subj in subjects:
        for label in labels:
            for i in range(per_combo):
                rows.append({
                    "filepath": f"data/raw/leapgestrecog/{subj}/{label}/{i}.png",
                    "label": label,
                    "subject": f"leap_{subj}",
                    "source": "leapgestrecog",
                    "media_type": "image",
                })
    return rows


def _make_team_rows(count: int, label: str = "left_swipe") -> list[dict[str, str]]:
    """Generate fake manifest rows for team-recorded videos."""
    return [
        {
            "filepath": f"data/{label}/video_{i + 1:03d}.mp4",
            "label": label,
            "subject": "team",
            "source": "team_recorded",
            "media_type": "video",
        }
        for i in range(count)
    ]


# ---- Subject-grouped splitting tests ----


def test_split_by_subject_no_leakage() -> None:
    """No subject should appear in more than one split."""
    rows = _make_leapgest_rows(
        subjects=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
        labels=["palm", "fist"],
    )
    rng = random.Random(42)
    _split_by_subject(rows, rng)

    subject_splits: dict[str, set[str]] = {}
    for row in rows:
        subj = row["subject"]
        if subj not in subject_splits:
            subject_splits[subj] = set()
        subject_splits[subj].add(row["split"])

    for subj, splits in subject_splits.items():
        assert len(splits) == 1, f"Subject {subj} leaked across splits: {splits}"


def test_split_by_subject_every_row_gets_split() -> None:
    """Every row must have a valid split assignment."""
    rows = _make_leapgest_rows(
        subjects=["00", "01", "02", "03", "04"],
        labels=["palm", "fist", "thumb"],
    )
    rng = random.Random(42)
    _split_by_subject(rows, rng)

    for row in rows:
        assert row["split"] in {"train", "val", "test"}


def test_split_by_subject_all_three_splits_present() -> None:
    """With enough subjects, all three splits should have data."""
    rows = _make_leapgest_rows(
        subjects=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
        labels=["palm"],
    )
    rng = random.Random(42)
    _split_by_subject(rows, rng)

    splits_present = {row["split"] for row in rows}
    assert splits_present == {"train", "val", "test"}


def test_split_by_subject_is_reproducible() -> None:
    """Same seed should produce identical splits."""
    rows_a = _make_leapgest_rows(
        subjects=["00", "01", "02", "03", "04"],
        labels=["palm", "fist"],
    )
    rows_b = _make_leapgest_rows(
        subjects=["00", "01", "02", "03", "04"],
        labels=["palm", "fist"],
    )

    _split_by_subject(rows_a, random.Random(42))
    _split_by_subject(rows_b, random.Random(42))

    for a, b in zip(rows_a, rows_b):
        assert a["split"] == b["split"]


def test_split_by_subject_train_is_largest() -> None:
    """Training set should have the most data."""
    rows = _make_leapgest_rows(
        subjects=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
        labels=["palm", "fist"],
    )
    rng = random.Random(42)
    _split_by_subject(rows, rng)

    train_count = sum(1 for r in rows if r["split"] == "train")
    val_count = sum(1 for r in rows if r["split"] == "val")
    test_count = sum(1 for r in rows if r["split"] == "test")

    assert train_count > val_count
    assert train_count > test_count


# ---- Random splitting tests ----


def test_split_randomly_every_row_gets_split() -> None:
    """Every row must have a valid split assignment."""
    rows = _make_team_rows(10)
    rng = random.Random(42)
    _split_randomly(rows, rng)

    for row in rows:
        assert row["split"] in {"train", "val", "test"}


def test_split_randomly_is_reproducible() -> None:
    """Same seed should produce identical splits."""
    rows_a = _make_team_rows(10)
    rows_b = _make_team_rows(10)

    _split_randomly(rows_a, random.Random(42))
    _split_randomly(rows_b, random.Random(42))

    for a, b in zip(rows_a, rows_b):
        assert a["split"] == b["split"]


def test_split_randomly_handles_small_counts() -> None:
    """Even with very few samples, every row gets a split."""
    rows = _make_team_rows(2)
    rng = random.Random(42)
    _split_randomly(rows, rng)

    for row in rows:
        assert row["split"] in {"train", "val", "test"}


# ---- Error handling tests ----


def test_split_manifest_raises_on_missing_file(tmp_path: Path) -> None:
    """Should raise FileNotFoundError if manifest doesn't exist."""
    missing = tmp_path / "data" / "manifest.csv"
    output = tmp_path / "data" / "manifest_split.csv"
    with pytest.raises(FileNotFoundError):
        split_manifest(missing, output)


def test_split_manifest_raises_on_empty_manifest(tmp_path: Path) -> None:
    """Should raise ValueError if manifest is empty."""
    manifest = tmp_path / "data" / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    # Write header only, no data rows
    with open(manifest, "w", newline="") as f:
        f.write("filepath,label,subject,source,media_type\n")

    output = tmp_path / "data" / "manifest_split.csv"
    with pytest.raises(ValueError):
        split_manifest(manifest, output)
