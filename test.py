"""Main entry point for gesture-controlled YouTube.

Runs locally on your laptop — no Pi or socket needed.

Wires together:
    - cv2 webcam capture  (full frame for motion tracking)
    - GestureClassifier   (HMM-based swipe detection)
    - CommandEngine       (debounce + cooldown + keypress)
"""

import os
import signal
import sys
import time
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dynamic_classifier"))

from dynamic_classifier.inference import GestureClassifier
from command_engine.engine import CommandEngine, EngineConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
CLASSIFY_EVERY_N_FRAMES = 2

engine_config = EngineConfig(
    debounce_frames=15,
    cooldown_seconds=1.5,
    confidence_threshold=0.75,
    require_no_hand_reset=False,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: could not open webcam.")
    sys.exit(1)

classifier = GestureClassifier(window_size=20)
engine = CommandEngine(config=engine_config)

def _handle_sigint(sig, frame):
    print("\nShutting down.")
    cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_sigint)

print("Gesture control running. Press Ctrl+C to stop.")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

frame_index = 0
last_label = None
last_confidence = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to read frame.")
        break

    if frame_index % CLASSIFY_EVERY_N_FRAMES == 0:
        last_label, last_confidence = classifier.predict(frame)
        last_confidence = last_confidence or 0.0

    action = engine.process(last_label, confidence=last_confidence)
    if action:
        print(f"Action fired: {action}")

    frame_index += 1

cap.release()
