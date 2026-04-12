"""Main entry point for gesture-controlled YouTube on Raspberry Pi.

Wires together:
    - cv2 webcam capture  (full frame for motion tracking)
    - GestureClassifier   (HMM-based swipe detection)
    - CommandEngine       (debounce + cooldown + keypress)
    - Socket sender       (sends actions to laptop over local network)

Usage:
    1. Find your laptop's local IP (ipconfig / ifconfig)
    2. Set LAPTOP_IP below to that address
    3. Run listener.py on your laptop
    4. Run this script on the Pi
"""

import os
import signal
import socket
import sys
import time
import cv2

# Allow dynamic_classifier's internal imports (hmm, train, features) to resolve
# when main.py is run from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dynamic_classifier"))

from dynamic_classifier.inference import GestureClassifier
from command_engine.engine import CommandEngine, EngineConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAPTOP_IP   = "172.17.224.1"  # <-- replace with your laptop's local IP
PORT        = 5005

FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
CLASSIFY_EVERY_N_FRAMES = 2

engine_config = EngineConfig(
    debounce_frames=5,
    debounce_overrides={
        "right_swipe": 20,
        "left_swipe":  20,
    },
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

# Connect to laptop listener
print(f"Connecting to laptop at {LAPTOP_IP}:{PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((LAPTOP_IP, PORT))
print("Connected.")

def _handle_sigint(sig, frame):
    print("\nShutting down.")
    cap.release()
    sock.close()
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
        #print(f"Gesture: {last_label} ({last_confidence:.2f})")

    action = engine.process(last_label, confidence=last_confidence)
    if action:
        print(f"Action fired: {action}")
        sock.send(action.encode())

    frame_index += 1

cap.release()
sock.close()
