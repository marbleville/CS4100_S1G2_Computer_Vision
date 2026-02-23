"""Disk-backed writer for frame packets."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessor.io.types import FramePacket


class DiskFramePacketWriter:
    """Writes a source-agnostic FramePacket to disk."""

    def write_frame_packet(
        self,
        packet: FramePacket,
        output_dir: str | Path,
        file_name: str | None = None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stem = file_name or self._default_stem(packet)
        image_path = output_path / f"{stem}.png"
        meta_path = output_path / f"{stem}.json"

        frame_rgb = np.asarray(packet.frame_rgb, dtype=np.uint8)
        image = Image.fromarray(frame_rgb, mode="RGB")
        image.save(image_path)

        metadata = {
            "frame_index": packet.frame_index,
            "timestamp_ms": packet.timestamp_ms,
            "source_id": packet.source_id,
            "height": int(frame_rgb.shape[0]),
            "width": int(frame_rgb.shape[1]),
            "channels": int(frame_rgb.shape[2]) if frame_rgb.ndim == 3 else 1,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return image_path

    @staticmethod
    def _default_stem(packet: FramePacket) -> str:
        safe_source = re.sub(r"[^a-zA-Z0-9_.-]+", "_",
                             packet.source_id).strip("_")
        safe_source = safe_source or "source"
        return f"{safe_source}_frame_{packet.frame_index:06d}_{packet.timestamp_ms}ms"
