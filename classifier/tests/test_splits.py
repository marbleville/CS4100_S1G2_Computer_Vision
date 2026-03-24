"""
Tests for the split loader utility.

Uses a synthetic manifest_split.csv to test loading, filtering,
label mapping, and edge cases without requiring real dataset files.

Tests are written against observable behavior only — they do not
assume knowledge of LABEL_MAP internals or split generation logic.
"""

import csv
import os

import pytest

from classifier.data.splits import load_splits, get_split_paths, print_split_summary
from classifier.config import STATIC_GESTURE_CLASSES


# Helpers
def write_manifest(path: str, rows: list[dict]) -> None:
    """Write a manifest CSV with Module A's column schema."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["filepath", "label", "subject", "source", "media_type", "split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def image_row(
    label: str,
    split: str,
    index: int = 0,
    source: str = "leapgestrecog",
) -> dict:
    """Build a single image row as Module A would write it."""
    return {
        "filepath": f"data/gestures/{label}/img_{index:03d}.jpg",
        "label": label,
        "subject": f"leap_0{index % 5}",
        "source": source,
        "media_type": "image",
        "split": split,
    }

def video_row(label: str, split: str, index: int = 0) -> dict:
    """Build a single video row as Module A would write it."""
    return {
        "filepath": f"data/{label}/video_{index:03d}.mov",
        "label": label,
        "subject": "team",
        "source": "team_recorded",
        "media_type": "video",
        "split": split,
    }

# Fixtures
@pytest.fixture
def manifest_with_known_mappings(tmp_path):
    """
    Manifest using only the four raw labels we know map to static gestures.
    Explicitly assigns known splits so tests can assert exact membership.
    """
    path = str(tmp_path / "data" / "manifest_split.csv")
    rows = [
        # fist -> fist
        image_row("fist", "train", 0),
        image_row("fist", "train", 1),
        image_row("fist", "val",   2),
        image_row("fist", "test",  3),
        # palm -> open
        image_row("palm", "train", 0),
        image_row("palm", "train", 1),
        image_row("palm", "val",   2),
        image_row("palm", "test",  3),
        # thumb -> thumbs_up
        image_row("thumb", "train", 0),
        image_row("thumb", "val",   1),
        image_row("thumb", "test",  2),
        # down -> thumbs_down
        image_row("down", "train", 0),
        image_row("down", "val",   1),
        image_row("down", "test",  2),
    ]
    write_manifest(path, rows)
    return path

# Basic structure tests
def test_returns_dict_with_three_split_keys(manifest_with_known_mappings):
    splits = load_splits(manifest_with_known_mappings)
    assert set(splits.keys()) == {"train", "val", "test"}

def test_each_record_has_non_empty_path_and_label(manifest_with_known_mappings):
    """Records must have non-empty string values, not just the keys."""
    splits = load_splits(manifest_with_known_mappings)
    for split_name, records in splits.items():
        for record in records:
            assert isinstance(record.get("path"), str) and len(record["path"]) > 0, \
                f"Empty path in {split_name}"
            assert isinstance(record.get("label"), str) and len(record["label"]) > 0, \
                f"Empty label in {split_name}"

def test_all_output_labels_are_valid_gesture_classes(manifest_with_known_mappings):
    """
    Every label in the output must be one of our canonical gesture classes.
    This tests the mapping result without knowing the map internals.
    """
    splits = load_splits(manifest_with_known_mappings)
    for split_name, records in splits.items():
        for record in records:
            assert record["label"] in STATIC_GESTURE_CLASSES, (
                f"Unexpected label '{record['label']}' in {split_name} — "
                f"must be one of {STATIC_GESTURE_CLASSES}"
            )

def test_no_raw_module_a_labels_in_output(manifest_with_known_mappings):
    """
    Raw Module A labels like 'palm', 'thumb', 'down' must not appear in output.
    Output must only contain our canonical labels.
    """
    splits = load_splits(manifest_with_known_mappings)
    raw_labels = {"palm", "thumb", "down", "fist_moved", "palm_moved", "l", "ok", "c", "index"}
    for split_name, records in splits.items():
        for record in records:
            assert record["label"] not in raw_labels, (
                f"Raw Module A label '{record['label']}' found in output — "
                f"label mapping is not being applied"
            )

# Split assignment tests
def test_train_rows_land_in_train(manifest_with_known_mappings):
    """Rows with split='train' in the CSV should appear in the train split."""
    splits = load_splits(manifest_with_known_mappings)
    assert len(splits["train"]) > 0, "Train split is empty"

def test_val_rows_land_in_val(manifest_with_known_mappings):
    splits = load_splits(manifest_with_known_mappings)
    assert len(splits["val"]) > 0, "Val split is empty"

def test_test_rows_land_in_test(manifest_with_known_mappings):
    splits = load_splits(manifest_with_known_mappings)
    assert len(splits["test"]) > 0, "Test split is empty"

def test_no_path_appears_in_multiple_splits(manifest_with_known_mappings):
    """
    Each image path must appear in exactly one split.
    Overlap would mean the same image is used for both training and evaluation.
    """
    splits = load_splits(manifest_with_known_mappings)
    train_paths = {r["path"] for r in splits["train"]}
    val_paths   = {r["path"] for r in splits["val"]}
    test_paths  = {r["path"] for r in splits["test"]}
    assert len(train_paths & val_paths) == 0,  "Same image in train and val"
    assert len(train_paths & test_paths) == 0, "Same image in train and test"
    assert len(val_paths & test_paths) == 0,   "Same image in val and test"

# Filtering tests
def test_video_rows_never_appear_in_output(tmp_path):
    """
    Video rows must be excluded regardless of their label or split.
    Videos are Module D's responsibility.
    """
    path = str(tmp_path / "data" / "manifest_split.csv")
    rows = [
        image_row("fist", "train", 0),
        video_row("left_swipe",  "train", 0),
        video_row("right_swipe", "train", 1),
    ]
    write_manifest(path, rows)
    splits = load_splits(path)
    all_paths = [r["path"] for split in splits.values() for r in split]
    assert not any(".mov" in p or ".mp4" in p for p in all_paths), \
        "Video file paths found in output — videos should be excluded"

def test_unknown_labels_do_not_appear_in_output(tmp_path):
    """
    Labels with no mapping should be silently dropped, not crash or pass through.
    """
    path = str(tmp_path / "data" / "manifest_split.csv")
    rows = [
        image_row("fist", "train", 0),
        image_row("fist", "train", 1),
        {
            "filepath": "data/gestures/totally_unknown/img_000.jpg",
            "label": "totally_unknown",
            "subject": "leap_00",
            "source": "leapgestrecog",
            "media_type": "image",
            "split": "train",
        },
    ]
    write_manifest(path, rows)
    splits = load_splits(path)
    all_labels = {r["label"] for split in splits.values() for r in split}
    assert "totally_unknown" not in all_labels

def test_invalid_split_column_rows_are_dropped(tmp_path):
    """Rows with an invalid split value should be silently ignored."""
    path = str(tmp_path / "data" / "manifest_split.csv")
    rows = [
        image_row("fist", "train", 0),
        image_row("fist", "train", 1),
        {**image_row("fist", "train", 2), "split": "garbage_split"},
    ]
    write_manifest(path, rows)
    splits = load_splits(path)
    total = sum(len(v) for v in splits.values())
    # Only 2 valid rows should appear, the garbage_split row should be dropped
    assert total == 2, f"Expected 2 valid rows, got {total}"

# Error handling tests
def test_missing_manifest_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_splits("nonexistent/path/manifest_split.csv")

def test_all_rows_filtered_raises_value_error(tmp_path):
    """
    If every row is filtered out (e.g. all videos, no static images),
    a ValueError should be raised rather than returning empty splits silently.
    """
    path = str(tmp_path / "data" / "manifest_split.csv")
    rows = [
        video_row("left_swipe",  "train", 0),
        video_row("right_swipe", "train", 1),
    ]
    write_manifest(path, rows)
    with pytest.raises(ValueError):
        load_splits(path)

# Helper function tests
def test_get_split_paths_returns_list_of_two_tuples(manifest_with_known_mappings):
    splits = load_splits(manifest_with_known_mappings)
    paths = get_split_paths(splits, "train")
    assert isinstance(paths, list)
    assert len(paths) > 0
    for item in paths:
        assert isinstance(item, tuple) and len(item) == 2, \
            f"Expected (path, label) tuple, got {item}"

def test_get_split_paths_labels_match_canonical(manifest_with_known_mappings):
    """Labels returned by get_split_paths must be canonical, not raw Module A labels."""
    splits = load_splits(manifest_with_known_mappings)
    for split_name in ("train", "val", "test"):
        for path, label in get_split_paths(splits, split_name):
            assert label in STATIC_GESTURE_CLASSES, \
                f"Non-canonical label '{label}' returned by get_split_paths"

def test_get_split_paths_invalid_name_raises(manifest_with_known_mappings):
    splits = load_splits(manifest_with_known_mappings)
    with pytest.raises(ValueError):
        get_split_paths(splits, "not_a_real_split")

def test_print_split_summary_mentions_all_splits(manifest_with_known_mappings, capsys):
    splits = load_splits(manifest_with_known_mappings)
    print_split_summary(splits)
    captured = capsys.readouterr()
    assert "train" in captured.out
    assert "val" in captured.out
    assert "test" in captured.out