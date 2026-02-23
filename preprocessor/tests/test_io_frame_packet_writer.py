from __future__ import annotations

import json

import numpy as np
from PIL import Image

from preprocessor.io.frame_packet_writer import DiskFramePacketWriter
from preprocessor.io.types import FramePacket


def _sample_packet() -> FramePacket:
    frame = np.full((6, 8, 3), 127, dtype=np.uint8)
    return FramePacket(
        frame_index=3,
        timestamp_ms=150,
        frame_rgb=frame,
        source_id="sample source.mov",
    )


def test_write_frame_packet_creates_image_and_metadata(tmp_path) -> None:
    writer = DiskFramePacketWriter()
    packet = _sample_packet()

    image_path = writer.write_frame_packet(packet, tmp_path)
    metadata_path = image_path.with_suffix(".json")

    assert image_path.exists()
    assert image_path.suffix == ".png"
    assert metadata_path.exists()

    image = Image.open(image_path)
    assert image.mode == "RGB"
    assert image.size == (8, 6)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["frame_index"] == 3
    assert metadata["timestamp_ms"] == 150
    assert metadata["source_id"] == "sample source.mov"
    assert metadata["height"] == 6
    assert metadata["width"] == 8
    assert metadata["channels"] == 3


def test_write_frame_packet_respects_custom_file_name(tmp_path) -> None:
    writer = DiskFramePacketWriter()
    packet = _sample_packet()

    image_path = writer.write_frame_packet(packet, tmp_path, file_name="manual_name")

    assert image_path.name == "manual_name.png"
    assert image_path.with_suffix(".json").name == "manual_name.json"
