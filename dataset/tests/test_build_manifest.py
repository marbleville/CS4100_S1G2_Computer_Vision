"""Tests for dataset manifest building."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset.build_manifest import (
    LEAPGESTRECOG_LABELS,
    SWIPE_LABELS,
    build_manifest,
    _scan_leapgestrecog,
    _scan_swipe_videos,
)
from dataset.gesture_map import STATIC_GESTURES, DYNAMIC_GESTURES, ACTIVE_LABELS


def _create_fake_leapgestrecog(tmp_path: Path) -> Path:
    """Create a minimal fake LeapGestRecog folder structure.

    Includes both active gestures (palm, fist) and inactive ones
    (02_l, 07_ok) to verify filtering works.
    """
    leap_dir = tmp_path / "data" / "raw" / "leapgestrecog"
    for subject in ["00", "01"]:
        # Active gestures — should be included
        for folder_name in ["01_palm", "03_fist"]:
            gesture_dir = leap_dir / subject / folder_name
            gesture_dir.mkdir(parents=True)
            for i in range(3):
                (gesture_dir / f"frame_{i:05d}.png").write_text("fake")
        # Inactive gestures — should be skipped
        for folder_name in ["02_l", "07_ok"]:
            gesture_dir = leap_dir / subject / folder_name
            gesture_dir.mkdir(parents=True)
            for i in range(3):
                (gesture_dir / f"frame_{i:05d}.png").write_text("fake")
    return leap_dir


def _create_fake_swipe_dir(tmp_path: Path, label: str) -> Path:
    """Create a minimal fake swipe video folder."""
    swipe_dir = tmp_path / "data" / label
    swipe_dir.mkdir(parents=True)
    for i in range(2):
        (swipe_dir / f"video_{i + 1:03d}.mp4").write_text("fake")
    return swipe_dir


def test_scan_leapgestrecog_only_includes_active_gestures(tmp_path: Path) -> None:
    """Should only include gestures listed in STATIC_GESTURES."""
    leap_dir = _create_fake_leapgestrecog(tmp_path)
    rows = _scan_leapgestrecog(leap_dir, tmp_path)
    # 2 subjects × 2 active gestures × 3 images = 12
    # The 2 inactive gestures (l, ok) should be skipped
    assert len(rows) == 12
    labels = {r["label"] for r in rows}
    assert labels == {"palm", "fist"}
    assert "l" not in labels
    assert "ok" not in labels


def test_scan_leapgestrecog_labels_are_correct(tmp_path: Path) -> None:
    leap_dir = _create_fake_leapgestrecog(tmp_path)
    rows = _scan_leapgestrecog(leap_dir, tmp_path)
    labels = {r["label"] for r in rows}
    assert labels.issubset(STATIC_GESTURES.keys())


def test_scan_leapgestrecog_subjects_are_prefixed(tmp_path: Path) -> None:
    leap_dir = _create_fake_leapgestrecog(tmp_path)
    rows = _scan_leapgestrecog(leap_dir, tmp_path)
    subjects = {r["subject"] for r in rows}
    assert subjects == {"leap_00", "leap_01"}


def test_scan_leapgestrecog_all_rows_have_required_fields(tmp_path: Path) -> None:
    leap_dir = _create_fake_leapgestrecog(tmp_path)
    rows = _scan_leapgestrecog(leap_dir, tmp_path)
    required_fields = {"filepath", "label", "subject", "source", "media_type"}
    for row in rows:
        assert required_fields.issubset(row.keys())
        assert row["source"] == "leapgestrecog"
        assert row["media_type"] == "image"


def test_scan_swipe_videos_finds_all_videos(tmp_path: Path) -> None:
    swipe_dir = _create_fake_swipe_dir(tmp_path, "left_swipe")
    rows = _scan_swipe_videos(swipe_dir, "left_swipe", tmp_path)
    assert len(rows) == 2


def test_scan_swipe_videos_labels_are_correct(tmp_path: Path) -> None:
    swipe_dir = _create_fake_swipe_dir(tmp_path, "right_swipe")
    rows = _scan_swipe_videos(swipe_dir, "right_swipe", tmp_path)
    for row in rows:
        assert row["label"] == "right_swipe"
        assert row["source"] == "team_recorded"
        assert row["media_type"] == "video"


def test_scan_ignores_non_media_files(tmp_path: Path) -> None:
    swipe_dir = tmp_path / "data" / "left_swipe"
    swipe_dir.mkdir(parents=True)
    (swipe_dir / "video_001.mp4").write_text("fake")
    (swipe_dir / "notes.txt").write_text("not a video")
    (swipe_dir / ".DS_Store").write_text("system file")
    rows = _scan_swipe_videos(swipe_dir, "left_swipe", tmp_path)
    assert len(rows) == 1


def test_leapgestrecog_labels_only_contains_active_gestures() -> None:
    """The filtered label map should only have gestures from STATIC_GESTURES."""
    for folder, label in LEAPGESTRECOG_LABELS.items():
        assert label in STATIC_GESTURES, f"{label} is not an active gesture"


def test_active_labels_matches_gesture_counts() -> None:
    """Should have 4 static + 2 dynamic = 6 active gesture labels."""
    assert len(STATIC_GESTURES) == 4
    assert len(DYNAMIC_GESTURES) == 2
    assert len(ACTIVE_LABELS) == 6


def test_build_manifest_raises_on_no_data(tmp_path: Path) -> None:
    """Should raise FileNotFoundError when no data sources exist."""
    output_path = tmp_path / "data" / "manifest.csv"
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path, output_path)
