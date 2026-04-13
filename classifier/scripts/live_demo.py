"""
Live gesture recognition demo.

Streams webcam frames through Module B's preprocessing pipeline,
passes candidates through the gesture classifier, and prints
predictions in real time with FPS tracking.

Usage:
    python3 classifier/scripts/live_demo.py

Controls:
    Ctrl+C to exit cleanly.
"""

import time
import sys
from collections import deque

from preprocessor import init_preprocessor
from preprocessor.config.types import PreprocessorConfig

from classifier.static_classifier import StaticClassifier
from classifier.data.adapter import candidate_to_detection
from classifier.config import CONFIDENCE_THRESHOLD


# Configuration                                                       

CHECKPOINT_PATH = "classifier/models/weights/cnn_best.pt"
FPS_WINDOW = 30       # number of recent frames to average FPS over
MIN_CONFIDENCE = CONFIDENCE_THRESHOLD


def main() -> None:
    print("=" * 60)
    print("Live Gesture Recognition Demo")
    print("=" * 60)
    print(f"Confidence threshold: {MIN_CONFIDENCE}")
    print("Press Ctrl+C to exit\n")

    # Initialize preprocessor with webcam
    config = PreprocessorConfig(input_mode="webcam")
    preprocessor = init_preprocessor(config)

    # Give camera time to warm up
    import time
    time.sleep(2)

    # Load classifier
    print("Loading classifier...")
    classifier = StaticClassifier(model_path=CHECKPOINT_PATH)
    print("Classifier ready.\n")

    # FPS tracking
    frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
    total_candidates = 0
    total_predictions = 0
    last_gesture = None

    print(f"{'Frame':>8} {'FPS':>6} {'Gesture':>14} {'Confidence':>12} {'Status':>10}")
    print("-" * 56)

    try:
        frame_idx = 0
        while True:
            t_start = time.perf_counter()

            # Get next candidate from Module B
            candidate = preprocessor.next()

            if candidate is None:
                continue

            # Only use the primary candidate per frame (largest region)
            # Secondary candidates are likely face/arm not the hand
            if candidate.candidate_index != 0:
                continue

            total_candidates += 1

            # Convert and classify
            detection = candidate_to_detection(candidate)
            result = classifier.classify(detection)

            t_end = time.perf_counter()
            frame_times.append(t_end - t_start)

            # Compute FPS
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            # Determine status label
            if not result.hand_detected:
                status = "no hand"
            elif result.gesture is None:
                status = "low conf"
            else:
                status = "OK"
                total_predictions += 1

            # Only print when gesture changes or every 10 frames to avoid spam
            gesture_changed = result.gesture != last_gesture
            if gesture_changed or frame_idx % 10 == 0:
                gesture_str = result.gesture or "---"
                print(
                    f"{frame_idx:>8} "
                    f"{fps:>6.1f} "
                    f"{gesture_str:>14} "
                    f"{result.confidence:>11.3f} "
                    f"{status:>10}"
                )
                last_gesture = result.gesture

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n" + "=" * 56)
        print("Demo stopped.")
        print(f"Total candidates processed: {total_candidates}")
        print(f"Total confident predictions: {total_predictions}")
        if frame_times:
            avg_fps = len(frame_times) / sum(frame_times)
            print(f"Average FPS (last {FPS_WINDOW} frames): {avg_fps:.1f}")
        print("=" * 56)
        sys.exit(0)


if __name__ == "__main__":
    main()