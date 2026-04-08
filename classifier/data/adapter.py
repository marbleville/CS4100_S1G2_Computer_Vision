"""
Adapter layer between Module B's HandCandidateFrame and Module C's HandDetectionResult.

Module B does not guarantee a hand is present — it outputs candidate regions that are
likely to be part of a body. Module C's CNN confidence threshold acts as the hand
confirmation step. Any candidate below the confidence threshold returns gesture=None.
"""

import numpy as np
from PIL import Image

from preprocessor.pipeline.types import HandCandidateFrame
from preprocessor.types.results import HandDetectionResult

TARGET_CROP_SIZE = 128


def candidate_to_detection(candidate: HandCandidateFrame) -> HandDetectionResult:
    """
    Convert a HandCandidateFrame from Module B into a HandDetectionResult for Module C.

    Resizes the crop to 128x128 if Module B's candidate_frame_size_px differs.
    Sets hand_detected=True and confidence_level=1.0 since Module B does not provide
    a confidence score — the CNN threshold in StaticClassifier handles filtering.

    Args:
        candidate: HandCandidateFrame from Module B's preprocessing pipeline.

    Returns:
        HandDetectionResult ready for StaticClassifier.classify().
    """
    crop = candidate.frame_rgb

    # Resize if Module B crop size does not match CNN input size
    if crop.shape[0] != TARGET_CROP_SIZE or crop.shape[1] != TARGET_CROP_SIZE:
        pil = Image.fromarray(crop).resize(
            (TARGET_CROP_SIZE, TARGET_CROP_SIZE), Image.Resampling.BILINEAR
        )
        crop = np.asarray(pil, dtype=np.uint8)

    return HandDetectionResult(
        hand_detected=True,
        confidence_level=1.0,
        crop_rgb=crop,
        timestamp_ms=candidate.timestamp_ms,
        bbox=candidate.bbox_xyxy_px,
    )