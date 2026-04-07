"""Gesture-to-action mapping for the project.

This is the single source of truth for which gestures the system
recognizes and what media actions they trigger. All other scripts
(manifest builder, classifier, command engine) should reference
this file.

MVP gesture set (7 total):
    4 static gestures  (from LeapGestRecog dataset)
    2 dynamic gestures (team-recorded swipe videos)
    1 implicit state   (no hand detected)
"""

# Static gestures: single-frame hand poses from LeapGestRecog
STATIC_GESTURES = {
    "palm":  "play_pause",    # open hand  → toggle play/pause (spacebar)
    "thumb": "volume_up",     # thumbs up  → increase volume (up arrow)
    "down":  "volume_down",   # point down → decrease volume (down arrow)
    "fist":  "mute",          # closed fist → mute/unmute (m)
}

# Dynamic gestures: motion-based from team-recorded videos
DYNAMIC_GESTURES = {
    "right_swipe": "next_track",      # swipe right → next video/track (shift + n)
    "left_swipe":  "previous_track",  # swipe left  → previous video/track (shift + p)
}

# All active gesture labels (used by manifest builder to filter)
ACTIVE_LABELS = set(STATIC_GESTURES.keys()) | set(DYNAMIC_GESTURES.keys())

# Complete mapping for reference
ALL_GESTURE_ACTIONS = {**STATIC_GESTURES, **DYNAMIC_GESTURES}
