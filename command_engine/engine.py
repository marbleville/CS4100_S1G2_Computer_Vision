"""Command engine that maps gesture predictions to keyboard actions.

This module sits between the classifier and the OS. It receives
gesture predictions frame-by-frame and decides when to actually
press a key, using safety logic to prevent accidental or repeated
triggers."""

import time
from dataclasses import dataclass
from typing import Callable

from dataset.gesture_map import ALL_GESTURE_ACTIONS


# Map action names to keyboard keys (YouTube shortcuts).
#
# Values are either a single key string or a tuple of keys
# for combo presses (e.g., ("shift", "n") for Shift+N).
#
# YouTube-specific mappings:
#   - K (works even if video isn't in focus) / Space  → play/pause  
#   - Up arrow   → volume +5%
#   - Down arrow → volume -5%
#   - M          → mute/unmute
#   - Shift+N    → next video 
#   - Shift+P    → previous video 

ACTION_TO_KEY: dict[str, str | tuple[str, ...]] = {
    "play_pause":     "k",
    "volume_up":      "up",
    "volume_down":    "down",
    "mute":           "m",
    "next_track":     ("shift", "n"),
    "previous_track": ("shift", "p"),
}


def _default_keypress(key: str | tuple[str, ...]) -> None:
    """Press a keyboard key or key combo using pyautogui.

    Accepts either a single key string (e.g., "k") or a tuple
    of keys for combos (e.g., ("shift", "n") for Shift+N).

    Separated into its own function so tests can inject a mock
    without needing a display server or pyautogui installed.
    """
    import pyautogui
    pyautogui.PAUSE = 0
    if isinstance(key, tuple):
        pyautogui.hotkey(*key)
    else:
        pyautogui.press(key)


@dataclass
class EngineConfig:
    """Tunable parameters for the command engine."""

    debounce_frames: int = 5
    cooldown_seconds: float = 1.5
    confidence_threshold: float = 0.75
    require_no_hand_reset: bool = True


class CommandEngine:
    """Stateful engine that converts gesture predictions to key presses.

    Call process() once per frame with the classifier's output.
    The engine handles all safety logic internally.

    Args:
        config: Tunable engine parameters.
        keypress_fn: Function to press a key. Defaults to pyautogui.
            Pass a mock in tests.
        time_fn: Function returning current time in seconds.
            Defaults to time.monotonic. Pass a mock in tests
            to control cooldown timing.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        keypress_fn: Callable[[str | tuple[str, ...]], None] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._config = config or EngineConfig()
        self._keypress_fn = keypress_fn or _default_keypress
        self._time_fn = time_fn or time.monotonic

        # Debounce state
        # Tracks how many consecutive frames we've seen the same gesture.
        # Resets when the gesture changes, when NO_HAND is detected,
        # or when confidence drops below threshold.
        self._streak_gesture: str | None = None
        self._streak_count: int = 0

        # Cooldown state
        # Records the timestamp of the last fire for each gesture.
        self._last_fired: dict[str, float] = {}

        # No-hand reset state
        # After a gesture fires, it's marked as "needs reset."
        # It can only fire again after a NO_HAND frame clears this flag.
        self._needs_reset: dict[str, bool] = {}

    def process(
        self,
        gesture_label: str | None,
        confidence: float = 1.0,
    ) -> str | None:
        """Process one frame's gesture prediction.

        Args:
            gesture_label: Predicted gesture (e.g., "palm", "fist"),
                or None if no hand detected / preprocessor returned
                NO_HAND.
            confidence: Classifier confidence between 0.0 and 1.0.

        Returns:
            Action name that was triggered (e.g., "play_pause"),
            or None if no command was fired this frame.
        """
        # NO_HAND or low confidence: reset debounce, clear reset flags
        if gesture_label is None or confidence < self._config.confidence_threshold:
            self._streak_gesture = None
            self._streak_count = 0
            # A NO_HAND frame means the user lowered their hand,
            # so clear the "needs reset" flag for all gestures.
            for gesture in self._needs_reset:
                self._needs_reset[gesture] = False
            return None

        # Unknown gesture: not in our mapping, ignore
        action = ALL_GESTURE_ACTIONS.get(gesture_label)
        if action is None:
            self._streak_gesture = None
            self._streak_count = 0
            return None

        # Debounce: count consecutive frames of the same gesture
        if gesture_label == self._streak_gesture:
            self._streak_count += 1
        else:
            self._streak_gesture = gesture_label
            self._streak_count = 1

        # Not enough consecutive frames yet
        if self._streak_count < self._config.debounce_frames:
            return None

        # Cooldown: don't fire if we fired too recently
        now = self._time_fn()
        last_fire = self._last_fired.get(gesture_label, 0.0)
        if now - last_fire < self._config.cooldown_seconds:
            return None

        # No-hand reset: don't fire again until hand was lowered
        if self._config.require_no_hand_reset:
            if self._needs_reset.get(gesture_label, False):
                return None

        # All checks passed: fire the command
        key = ACTION_TO_KEY.get(action)
        if key is None:
            raise ValueError(
                f"No keyboard mapping for action '{action}'. "
                f"Add it to ACTION_TO_KEY in command_engine/engine.py"
            )

        self._keypress_fn(key)

        # Update state after firing
        self._last_fired[gesture_label] = now
        self._needs_reset[gesture_label] = True
        self._streak_count = 0  # Reset so it doesn't fire next frame

        return action

    def reset(self) -> None:
        """Reset all internal state.

        Call when restarting the recognition loop, switching
        video players, or starting a new session.
        """
        self._streak_gesture = None
        self._streak_count = 0
        self._last_fired.clear()
        self._needs_reset.clear()
