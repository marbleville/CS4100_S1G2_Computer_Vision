"""Tests for dataset split verification."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dataset.verify_split import verify_split


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a test manifest CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "label", "subject", "source", "media_type", "split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_valid_manifest(tmp_path: Path) -> list[dict[str, str]]:
    """Create valid manifest rows with real files on disk."""
    rows = []
    for split, subjects in [("train", ["00", "01"]), ("val", ["02"]), ("test", ["03"])]:
        for subj in subjects:
            for label in ["palm", "fist"]:
                for i in range(3):
                    rel_path = f"data/raw/leapgestrecog/{subj}/{label}/{i}.png"
                    full_path = tmp_path / rel_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text("fake image")
                    rows.append({
                        "filepath": rel_path,
                        "label": label,
                        "subject": f"leap_{subj}",
                        "source": "leapgestrecog",
                        "media_type": "image",
                        "split": split,
                    })
    return rows


def test_verify_passes_on_valid_split(tmp_path: Path) -> None:
    """A properly split manifest should pass all checks."""
    rows = _make_valid_manifest(tmp_path)
    manifest_path = tmp_path / "data" / "manifest_split.csv"
    _write_manifest(manifest_path, rows)

    result = verify_split(manifest_path, tmp_path)
    assert result is True


def test_verify_raises_on_missing_manifest(tmp_path: Path) -> None:
    """Should raise FileNotFoundError if the manifest doesn't exist."""
    missing_path = tmp_path / "data" / "manifest_split.csv"
    with pytest.raises(FileNotFoundError):
        verify_split(missing_path, tmp_path)


def test_verify_raises_on_empty_manifest(tmp_path: Path) -> None:
    """Should raise ValueError if the manifest is empty."""
    manifest_path = tmp_path / "data" / "manifest_split.csv"
    manifest_path.parent.mkdir(parents=True)
    with open(manifest_path, "w", newline="") as f:
        f.write("filepath,label,subject,source,media_type,split\n")

    with pytest.raises(ValueError):
        verify_split(manifest_path, tmp_path)


def test_verify_detects_missing_split_assignment(tmp_path: Path) -> None:
    """Should fail if any row has no split value."""
    rows = _make_valid_manifest(tmp_path)
    rows[0]["split"] = ""  # blank split
    manifest_path = tmp_path / "data" / "manifest_split.csv"
    _write_manifest(manifest_path, rows)

    result = verify_split(manifest_path, tmp_path)
    assert result is False


def test_verify_detects_data_leakage(tmp_path: Path) -> None:
    """Should fail if a subject appears in multiple splits."""
    rows = _make_valid_manifest(tmp_path)
    # Put a subject that's in train also into test
    for row in rows:
        if row["subject"] == "leap_00":
            row["split"] = "test"
            break
    # Keep the rest of leap_00 in train — now leap_00 is in both
    manifest_path = tmp_path / "data" / "manifest_split.csv"
    _write_manifest(manifest_path, rows)

    result = verify_split(manifest_path, tmp_path)
    assert result is False


def test_verify_detects_missing_files(tmp_path: Path) -> None:
    """Should fail if referenced files don't exist on disk."""
    rows = _make_valid_manifest(tmp_path)
    # Add a row pointing to a file that doesn't exist
    rows.append({
        "filepath": "data/raw/leapgestrecog/99/palm/ghost.png",
        "label": "palm",
        "subject": "leap_99",
        "source": "leapgestrecog",
        "media_type": "image",
        "split": "test",
    })
    manifest_path = tmp_path / "data" / "manifest_split.csv"
    _write_manifest(manifest_path, rows)

    result = verify_split(manifest_path, tmp_path)
    assert result is False
