"""Common frame source interface."""

from typing import Protocol

from preprocessor.io.types import FramePacket


class FrameSource(Protocol):
    """Common pull-based frame source abstraction."""

    def open(self) -> None:
        """Open underlying resources for reading frames."""

    def read(self) -> FramePacket | None:
        """Read one frame packet, or `None` if end-of-stream."""

    def close(self) -> None:
        """Close underlying resources."""
