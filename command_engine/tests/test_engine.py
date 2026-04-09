"""Tests for the command engine.

These tests simulate the real-time loop where the preprocessor
feeds hand detection results and the classifier provides gesture
predictions. The engine is tested with injectable mocks for both
keypress and timing, so no display or pyautogui is needed.
"""

from __future__ import annotations

from command_engine.engine import CommandEngine, EngineConfig, ACTION_TO_KEY


class MockKeypress:
    """Records keypresses instead of actually pressing keys."""

    def __init__(self):
        self.pressed: list[str] = []

    def __call__(self, key: str) -> None:
        self.pressed.append(key)


class MockClock:
    """Controllable clock for testing cooldown logic."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_engine(mock_keys: MockKeypress, clock: MockClock | None = None, **kwargs) -> CommandEngine:
    """Create an engine with test-friendly defaults."""
    defaults = {
        "debounce_frames": 3,
        "cooldown_seconds": 1.0,
        "confidence_threshold": 0.75,
        "require_no_hand_reset": True,
    }
    defaults.update(kwargs)
    return CommandEngine(
        config=EngineConfig(**defaults),
        keypress_fn=mock_keys,
        time_fn=clock or MockClock(),
    )


# Debounce tests
# These verify that the engine requires multiple consecutive frames
# of the same gesture before firing, preventing flickering.


def test_debounce_requires_consecutive_frames() -> None:
    """Command should not fire until debounce threshold is met."""
    keys = MockKeypress()
    engine = _make_engine(keys, debounce_frames=3)

    # Frames 1 and 2: not enough yet
    assert engine.process("palm", confidence=0.9) is None
    assert engine.process("palm", confidence=0.9) is None

    # Frame 3: threshold met, fires
    result = engine.process("palm", confidence=0.9)
    assert result == "play_pause"
    assert keys.pressed == ["k"]


def test_debounce_resets_on_gesture_change() -> None:
    """Switching gestures should reset the debounce counter."""
    keys = MockKeypress()
    engine = _make_engine(keys, debounce_frames=3)

    engine.process("palm", confidence=0.9)  # palm streak: 1
    engine.process("palm", confidence=0.9)  # palm streak: 2
    engine.process("fist", confidence=0.9)  # fist streak: 1 (palm reset)
    engine.process("fist", confidence=0.9)  # fist streak: 2
    result = engine.process("fist", confidence=0.9)  # fist streak: 3, fires

    assert result == "mute"
    assert keys.pressed == ["m"]


def test_debounce_resets_on_no_hand() -> None:
    """NO_HAND frame (gesture_label=None) should reset debounce.

    This mirrors the real-time loop where the preprocessor returns
    ResultStatus.NO_HAND and we pass None to the engine.
    """
    keys = MockKeypress()
    engine = _make_engine(keys, debounce_frames=3, require_no_hand_reset=False)

    engine.process("palm", confidence=0.9)  # streak: 1
    engine.process("palm", confidence=0.9)  # streak: 2
    engine.process(None)  # NO_HAND from preprocessor, resets streak
    engine.process("palm", confidence=0.9)  # streak: 1 (restarted)
    engine.process("palm", confidence=0.9)  # streak: 2

    assert len(keys.pressed) == 0  # Never reached 3


def test_debounce_resets_on_low_confidence() -> None:
    """Low-confidence frame should be treated same as NO_HAND.

    If the classifier isn't sure what it's seeing, we don't want
    to count that frame toward the debounce threshold.
    """
    keys = MockKeypress()
    engine = _make_engine(keys, debounce_frames=3, confidence_threshold=0.8)

    engine.process("palm", confidence=0.9)  # streak: 1
    engine.process("palm", confidence=0.9)  # streak: 2
    engine.process("palm", confidence=0.5)  # low confidence, resets
    engine.process("palm", confidence=0.9)  # streak: 1 (restarted)
    engine.process("palm", confidence=0.9)  # streak: 2

    assert len(keys.pressed) == 0


# Confidence threshold tests


def test_low_confidence_never_fires() -> None:
    """Frames below confidence threshold should never trigger."""
    keys = MockKeypress()
    engine = _make_engine(keys, debounce_frames=1, confidence_threshold=0.8)

    engine.process("palm", confidence=0.5)
    engine.process("palm", confidence=0.5)
    engine.process("palm", confidence=0.5)

    assert len(keys.pressed) == 0


def test_high_confidence_fires() -> None:
    """Frames above confidence threshold should count normally."""
    keys = MockKeypress()
    engine = _make_engine(keys, debounce_frames=2, confidence_threshold=0.8)

    engine.process("palm", confidence=0.9)
    result = engine.process("palm", confidence=0.9)

    assert result == "play_pause"
    assert keys.pressed == ["k"]


# Cooldown tests
# These verify that the same gesture can't fire repeatedly in
# quick succession (e.g., holding palm up shouldn't spam spacebar).


def test_cooldown_prevents_rapid_fire() -> None:
    """Same gesture should not fire again within cooldown period."""
    keys = MockKeypress()
    clock = MockClock()
    engine = _make_engine(
        keys, clock,
        debounce_frames=1,
        cooldown_seconds=2.0,
        require_no_hand_reset=False,
    )

    # First fire at t=1000
    result = engine.process("palm", confidence=0.9)
    assert result == "play_pause"

    # Try again immediately — blocked by cooldown
    result = engine.process("palm", confidence=0.9)
    assert result is None

    # Advance past cooldown
    clock.advance(3.0)
    result = engine.process("palm", confidence=0.9)
    assert result == "play_pause"

    assert keys.pressed == ["k", "k"]


def test_cooldown_is_per_gesture() -> None:
    """Cooldown for one gesture shouldn't block a different gesture."""
    keys = MockKeypress()
    clock = MockClock()
    engine = _make_engine(
        keys, clock,
        debounce_frames=1,
        cooldown_seconds=5.0,
        require_no_hand_reset=False,
    )

    engine.process("palm", confidence=0.9)  # fires play_pause
    result = engine.process("fist", confidence=0.9)  # fires mute (different gesture)
    assert result == "mute"
    assert keys.pressed == ["k", "m"]


# No-hand reset tests
# These verify that the user must lower their hand (causing a
# NO_HAND frame from the preprocessor) before the same gesture
# can fire again. This is the key integration point with Module B.


def test_no_hand_reset_blocks_same_gesture() -> None:
    """Same gesture should not fire again until hand is lowered."""
    keys = MockKeypress()
    engine = _make_engine(
        keys,
        debounce_frames=1,
        cooldown_seconds=0.0,
        require_no_hand_reset=True,
    )

    result = engine.process("palm", confidence=0.9)
    assert result == "play_pause"

    # Keep holding palm — blocked by no-hand reset
    assert engine.process("palm", confidence=0.9) is None
    assert engine.process("palm", confidence=0.9) is None

    # Lower hand (preprocessor returns NO_HAND)
    engine.process(None)

    # Raise hand again — fires
    result = engine.process("palm", confidence=0.9)
    assert result == "play_pause"

    assert keys.pressed == ["k", "k"]


def test_no_hand_reset_only_blocks_fired_gesture() -> None:
    """Switching to a different gesture should work without lowering hand."""
    keys = MockKeypress()
    engine = _make_engine(
        keys,
        debounce_frames=1,
        cooldown_seconds=0.0,
        require_no_hand_reset=True,
    )

    engine.process("palm", confidence=0.9)  # fires play_pause
    result = engine.process("fist", confidence=0.9)  # fires mute
    assert result == "mute"
    assert keys.pressed == ["k", "m"]


def test_no_hand_reset_clears_on_none() -> None:
    """Passing None (NO_HAND) should clear reset flags for ALL gestures."""
    keys = MockKeypress()
    engine = _make_engine(
        keys,
        debounce_frames=1,
        cooldown_seconds=0.0,
        require_no_hand_reset=True,
    )

    engine.process("palm", confidence=0.9)  # fires, palm needs reset
    engine.process("fist", confidence=0.9)  # fires, fist needs reset

    # Both are now blocked
    assert engine.process("palm", confidence=0.9) is None
    assert engine.process("fist", confidence=0.9) is None

    # Lower hand — clears both
    engine.process(None)

    # Both can fire again
    assert engine.process("palm", confidence=0.9) == "play_pause"
    engine.process(None)  # reset again
    assert engine.process("fist", confidence=0.9) == "mute"


# Unknown gesture tests


def test_unknown_gesture_ignored() -> None:
    """Gestures not in gesture_map should be silently ignored."""
    keys = MockKeypress()
    engine = _make_engine(keys, debounce_frames=1)

    result = engine.process("some_random_gesture", confidence=0.9)
    assert result is None
    assert len(keys.pressed) == 0


# Reset tests


def test_reset_clears_all_state() -> None:
    """Engine reset should clear debounce, cooldown, and no-hand state."""
    keys = MockKeypress()
    clock = MockClock(start=1000.0)
    engine = _make_engine(
        keys, clock,
        debounce_frames=1,
        cooldown_seconds=999.0,
        require_no_hand_reset=True,
    )

    # First fire works (time=1000, last_fired=0, diff=1000 > 999)
    engine.process("palm", confidence=0.9)
    assert len(keys.pressed) == 1

    # Blocked by both cooldown and no-hand reset
    assert engine.process("palm", confidence=0.9) is None

    # Full reset
    engine.reset()

    # Should fire again
    result = engine.process("palm", confidence=0.9)
    assert result == "play_pause"
    assert len(keys.pressed) == 2


# Key mapping completeness


def test_all_gestures_have_key_mappings() -> None:
    """Every gesture in gesture_map should have a corresponding key."""
    from dataset.gesture_map import ALL_GESTURE_ACTIONS

    for gesture, action in ALL_GESTURE_ACTIONS.items():
        assert action in ACTION_TO_KEY, (
            f"Gesture '{gesture}' maps to action '{action}' "
            f"but '{action}' has no key in ACTION_TO_KEY"
        )


def test_all_static_gestures_fire_correct_keys() -> None:
    """Verify each static gesture produces the expected keypress."""
    expected = {
        "palm": "k",
        "thumb": "up",
        "down": "down",
        "fist": "m",
    }

    for gesture, expected_key in expected.items():
        keys = MockKeypress()
        engine = _make_engine(keys, debounce_frames=1)
        engine.process(gesture, confidence=0.9)
        assert keys.pressed == [expected_key], (
            f"Gesture '{gesture}' should press '{expected_key}' "
            f"but pressed {keys.pressed}"
        )


def test_dynamic_gestures_fire_key_combos() -> None:
    """Verify swipe gestures produce Shift+key combos for YouTube.

    YouTube uses Shift+N for next video and Shift+P for previous,
    so the engine must pass tuples (not single strings) to the
    keypress function.
    """
    expected = {
        "right_swipe": ("shift", "n"),
        "left_swipe":  ("shift", "p"),
    }

    for gesture, expected_combo in expected.items():
        keys = MockKeypress()
        engine = _make_engine(keys, debounce_frames=1)
        engine.process(gesture, confidence=0.9)
        assert keys.pressed == [expected_combo], (
            f"Gesture '{gesture}' should press {expected_combo} "
            f"but pressed {keys.pressed}"
        )
