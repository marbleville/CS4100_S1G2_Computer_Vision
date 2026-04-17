"""Main entry point for gesture-controlled YouTube.

Runs locally on laptop — no Pi or socket needed.

Wires together:
    - Preprocessor         (hand detection + cropping for static gestures)
    - StaticClassifier     (CNN-based static gesture recognition)
    - GestureClassifier    (HMM-based swipe detection)
    - CommandEngine        (debounce + cooldown + keypress)

Both classifiers share the same preprocessor — the static classifier
uses preprocessor.next() for cropped hand candidates, while the dynamic
classifier uses preprocessor.next_full_frame() for full frames needed
for centroid motion tracking. This avoids opening the webcam twice.
"""

import os
import signal
import sys
import time
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dynamic_classifier"))

from preprocessor import init_preprocessor
from preprocessor.config.types import PreprocessorConfig
from classifier.static_classifier import StaticClassifier
from classifier.data.adapter import candidate_to_detection
from dynamic_classifier.inference import GestureClassifier
from command_engine.engine import CommandEngine, EngineConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_PATH          = "classifier/models/weights/cnn_best.pt"
CLASSIFY_EVERY_N_FRAMES  = 2

engine_config = EngineConfig(
    debounce_frames=5,
    cooldown_seconds=1.5,
    confidence_threshold=0.75,
    require_no_hand_reset=False,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

preprocessor = init_preprocessor(PreprocessorConfig(input_mode="webcam"))
time.sleep(2)  # give camera time to warm up

static_classifier = StaticClassifier(model_path=CHECKPOINT_PATH)
dynamic_classifier = GestureClassifier(window_size=20)
engine = CommandEngine(config=engine_config)


def _handle_sigint(sig, frame):
    print("\nShutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_sigint)

print("Gesture control running. Press Ctrl+C to stop.")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

frame_index = 0
last_dynamic_label      = None
last_dynamic_confidence = 0.0

while True:
    # --- Dynamic classifier (HMM, full frame) ---
    packet = preprocessor.next_full_frame()
    if packet is None:
        time.sleep(0.01)
        continue

    # FramePacket gives RGB; GestureClassifier expects BGR
    frame_bgr = cv2.cvtColor(packet.frame_rgb, cv2.COLOR_RGB2BGR)

    if frame_index % CLASSIFY_EVERY_N_FRAMES == 0:
        last_dynamic_label, last_dynamic_confidence = dynamic_classifier.predict(frame_bgr)
        last_dynamic_confidence = last_dynamic_confidence or 0.0

    dynamic_action = engine.process(last_dynamic_label, confidence=last_dynamic_confidence)
    if dynamic_action:
        print(f"Dynamic action fired: {dynamic_action}")

    # --- Static classifier (CNN, cropped candidate) ---
    candidate = preprocessor.next()
    if candidate is not None and candidate.candidate_index == 0:
        detection = candidate_to_detection(candidate)
        result = static_classifier.classify(detection)
        if result.gesture is not None:
            print(f"Static gesture: {result.gesture} ({result.confidence:.2f})")
            static_action = engine.process(result.gesture, confidence=result.confidence)
            if static_action:
                print(f"Static action fired: {static_action}")

    frame_index += 1
