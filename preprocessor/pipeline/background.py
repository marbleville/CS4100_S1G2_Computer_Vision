"""Running-average background model for optional foreground cueing."""

from __future__ import annotations

import numpy as np


class RunningBackgroundModel:
    """Simple grayscale running-average background model."""

    def __init__(self, alpha: float = 0.08, warmup_frames: int = 8) -> None:
        self.alpha = float(alpha)
        self.warmup_frames = int(warmup_frames)
        self._background: np.ndarray | None = None
        self._seen_frames = 0

    def reset(self) -> None:
        self._background = None
        self._seen_frames = 0

    def update_and_score(self, gray: np.ndarray) -> tuple[np.ndarray, bool]:
        grayf = gray.astype(np.float32)
        if self._background is None:
            self._background = grayf.copy()
            self._seen_frames = 1
            return np.zeros_like(grayf, dtype=np.float32), True

        self._background = self.alpha * grayf + (1.0 - self.alpha) * self._background
        self._seen_frames += 1
        score = np.abs(grayf - self._background).astype(np.float32)
        in_warmup = self._seen_frames <= self.warmup_frames
        return score, in_warmup
